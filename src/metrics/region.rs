use crate::graph::Graph;
use ben::decode::{count_samples_from_file, BenDecoder};
use pbr::ProgressBar;
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

#[derive(Clone, Copy)]
pub enum RegionMetric {
    Splits,
    Pieces,
}

/// Count splits (regions spanning >1 district) or pieces (sum of district-set
/// sizes over all regions) for a single assignment against a single pre-loaded
/// region column.
///
/// Dense bitset keyed by interned region id × district id: one
/// `Vec<u64>` of length `n_regions * words_per_region`. `words_per_region`
/// is `ceil(n_districts / 64)` — for typical FL runs (< 64 districts) each
/// region occupies exactly one u64, so the whole bitset is `n_regions * 8`
/// bytes and sits in L1. Replaces the per-sample
/// `Vec<HashSet<u16>>` allocations from Phase 2.
fn region_metric_for_key(
    graph: &Graph,
    assignment: &[u16],
    region_col_idx: usize,
    metric: RegionMetric,
) -> u32 {
    let col = &graph.region_columns[region_col_idx];
    let n_regions = graph.region_id_counts[region_col_idx] as usize;
    if n_regions == 0 {
        return 0;
    }

    let max_d = assignment.iter().copied().max().unwrap_or(0) as usize;
    let words_per_region = (max_d / 64) + 1;
    let mut bitset = vec![0u64; n_regions * words_per_region];

    for (i, maybe_rid) in col.iter().enumerate() {
        if let Some(rid) = *maybe_rid {
            let d = assignment[i] as usize;
            let w = rid as usize * words_per_region + (d >> 6);
            bitset[w] |= 1u64 << (d & 63);
        }
    }

    match metric {
        RegionMetric::Splits => (0..n_regions)
            .filter(|&r| {
                let start = r * words_per_region;
                let popcount: u32 = bitset[start..start + words_per_region]
                    .iter()
                    .map(|w| w.count_ones())
                    .sum();
                popcount > 1
            })
            .count() as u32,
        RegionMetric::Pieces => (0..n_regions)
            .map(|r| {
                let start = r * words_per_region;
                bitset[start..start + words_per_region]
                    .iter()
                    .map(|w| w.count_ones())
                    .sum::<u32>()
            })
            .sum(),
    }
}

pub fn tally_and_save_region_metric(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    key_list: Vec<String>,
    metric: RegionMetric,
    show_progress: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Resolve keys → column indices once.
    let region_col_indices: Vec<usize> = key_list
        .iter()
        .map(|k| {
            *graph
                .region_index
                .get(k)
                .unwrap_or_else(|| panic!("region key {:?} not pre-loaded on graph", k))
        })
        .collect();

    let n_pb_tics = 100;
    let mut pb = if show_progress {
        Some(ProgressBar::new(n_pb_tics))
    } else {
        None
    };

    let ben_file = File::open(in_file_name).expect("BEN file not found");
    let decoder = BenDecoder::new(&ben_file).expect("Failed to initialize decoder");

    let basename = Path::new(in_file_name)
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy();
    eprintln!("Reading {:?}...", basename);

    let line_count = count_samples_from_file(Path::new(in_file_name), "ben")
        .expect("Failed to count samples in BEN file");

    let pb_step_size = (line_count / n_pb_tics as usize) as u32;
    let mut previous_step = 0;

    let mut sample_nums = Vec::with_capacity(line_count * key_list.len());
    let mut n_reps_nums = Vec::with_capacity(line_count * key_list.len());
    let mut accepted_nums = Vec::with_capacity(line_count * key_list.len());
    let mut metric_keys = Vec::with_capacity(line_count * key_list.len());
    let mut metric_values = Vec::with_capacity(line_count * key_list.len());

    let mut sample_count: u64 = 1;
    let mut accepted_count: u32 = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch: Vec<(Vec<u16>, u16)> = Vec::with_capacity(BATCH_SIZE);

    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, n_reps)) => {
                batch.push((assignment, n_reps));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, n_reps)| {
                            let counts: Vec<(String, u32)> = key_list
                                .iter()
                                .zip(region_col_indices.iter())
                                .map(|(key, &col_idx)| {
                                    (
                                        key.clone(),
                                        region_metric_for_key(
                                            &graph, assignment, col_idx, metric,
                                        ),
                                    )
                                })
                                .collect();
                            (*n_reps, counts)
                        })
                        .collect();

                    for (n_reps, counts_by_key) in results {
                        for (key, count_val) in counts_by_key {
                            sample_nums.push(sample_count);
                            n_reps_nums.push(n_reps as u32);
                            accepted_nums.push(accepted_count);
                            metric_keys.push(key);
                            metric_values.push(count_val);
                        }
                        sample_count += n_reps as u64;
                        accepted_count += 1;
                    }
                    batch.clear();
                }

                if show_progress && accepted_count - previous_step >= pb_step_size {
                    pb.as_mut().unwrap().inc();
                    previous_step = accepted_count;
                }
            }
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|(assignment, n_reps)| {
                let counts: Vec<(String, u32)> = key_list
                    .iter()
                    .zip(region_col_indices.iter())
                    .map(|(key, &col_idx)| {
                        (
                            key.clone(),
                            region_metric_for_key(&graph, assignment, col_idx, metric),
                        )
                    })
                    .collect();
                (*n_reps, counts)
            })
            .collect();
        for (n_reps, counts_by_key) in results {
            for (key, count_val) in counts_by_key {
                sample_nums.push(sample_count);
                n_reps_nums.push(n_reps as u32);
                accepted_nums.push(accepted_count);
                metric_keys.push(key);
                metric_values.push(count_val);
            }
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    if let Some(pb_ref) = pb.as_mut() {
        pb_ref.finish();
    }

    let metric_col_name = match metric {
        RegionMetric::Splits => "region_splits",
        RegionMetric::Pieces => "region_pieces",
    };

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_nums).into(),
        Series::new("n_reps".into(), n_reps_nums).into(),
        Series::new("accepted_count".into(), accepted_nums).into(),
        Series::new("region_key".into(), metric_keys).into(),
        Series::new(metric_col_name.into(), metric_values).into(),
    ])?;

    let mut file = File::create(out_file_name).unwrap_or_else(|_| {
        panic!(
            "Failed to create output file {:?}. The file may already exist.",
            out_file_name
        )
    });
    eprintln!("Writing final output...");
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(None))
        .finish(&mut df)?;

    eprintln!("Done!");
    Ok(())
}
