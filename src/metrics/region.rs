use crate::graph::Graph;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline};
use polars::prelude::*;
use std::fs::File;

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
    high_compression: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let region_col_indices: Vec<usize> = key_list
        .iter()
        .map(|k| {
            *graph
                .region_index
                .get(k)
                .unwrap_or_else(|| panic!("region key {:?} not pre-loaded on graph", k))
        })
        .collect();

    let total = count_samples(in_file_name)?;
    let cap = total * key_list.len();
    let mut sample_nums: Vec<u64> = Vec::with_capacity(cap);
    let mut n_reps_nums: Vec<u32> = Vec::with_capacity(cap);
    let mut accepted_nums: Vec<u32> = Vec::with_capacity(cap);
    let mut metric_keys: Vec<String> = Vec::with_capacity(cap);
    let mut metric_values: Vec<u32> = Vec::with_capacity(cap);

    run_pipeline(
        in_file_name,
        total,
        |assignment, _n_reps| {
            key_list
                .iter()
                .zip(region_col_indices.iter())
                .map(|(key, &col_idx)| {
                    (
                        key.clone(),
                        region_metric_for_key(&graph, assignment, col_idx, metric),
                    )
                })
                .collect::<Vec<(String, u32)>>()
        },
        |step, n_reps, accepted, counts| {
            for (key, count_val) in counts {
                sample_nums.push(step);
                n_reps_nums.push(n_reps);
                accepted_nums.push(accepted);
                metric_keys.push(key);
                metric_values.push(count_val);
            }
        },
        show_progress,
    )?;

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
        .with_compression(parquet_compression(high_compression))
        .finish(&mut df)?;

    eprintln!("Done!");
    Ok(())
}
