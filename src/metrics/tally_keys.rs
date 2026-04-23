use crate::graph::Graph;
use ben::decode::BenDecoder;
use pbr::ProgressBar;
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

/// Upper bound on district ids the dense-buffer path supports.
/// `observed` is a u128 bitmask indexed by district id.
const MAX_DISTRICTS: u16 = 128;

/// Per-sample tally result.
///
/// `totals` is a flat `Vec<f64>` of shape `[n_keys * n_districts]`, where
/// `n_districts = max(assignment) + 1`. `observed` has bit `d` set iff
/// district `d` appeared in this sample's assignment. Keeping `observed`
/// lets the writer emit columns only for districts that actually show up
/// somewhere in the ensemble (matching the pre-Phase-3 output shape).
struct TallyRow {
    sample_num: u64,
    n_reps: u32,
    accepted_count: u32,
    totals: Vec<f64>,
    n_districts: u16,
    observed: u128,
}

/// Hot loop: flat index into pre-parsed attribute columns, accumulate into a
/// flat per-district totals vector. No HashMap work inside the inner loop.
fn tally_keys(
    graph: &Graph,
    assignment: &[u16],
    attr_col_indices: &[usize],
) -> (Vec<f64>, u16, u128) {
    let mut observed: u128 = 0;
    let mut max_d: u16 = 0;
    for &d in assignment {
        if d >= MAX_DISTRICTS {
            panic!(
                "district id {} exceeds current {}-district limit; widen the observed bitmask",
                d, MAX_DISTRICTS
            );
        }
        observed |= 1u128 << d;
        if d > max_d {
            max_d = d;
        }
    }
    let n_districts = max_d as usize + 1;
    let n_keys = attr_col_indices.len();
    let mut totals = vec![0.0f64; n_keys * n_districts];
    for (k, &col_idx) in attr_col_indices.iter().enumerate() {
        let col = &graph.attr_columns[col_idx];
        let offset = k * n_districts;
        for (i, &v) in col.iter().enumerate() {
            totals[offset + assignment[i] as usize] += v;
        }
    }
    (totals, n_districts as u16, observed)
}

/// Bits set in `mask`, returned in ascending order.
fn sorted_district_ids(mut mask: u128) -> Vec<u16> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        out.push(mask.trailing_zeros() as u16);
        mask &= mask - 1;
    }
    out
}

fn save_tallies_to_parquet(
    file_path: &str,
    tallies: &[TallyRow],
    key_list: &[String],
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Global union of districts observed anywhere in the ensemble. Columns
    // come out in ascending district-id order, same as the BTreeMap fix
    // from Phase 0.
    let global_observed: u128 = tallies.iter().fold(0u128, |acc, t| acc | t.observed);
    let district_ids = sorted_district_ids(global_observed);

    let n_rows = tallies.len() * key_list.len();
    let mut sample_numbers: Vec<u64> = Vec::with_capacity(n_rows);
    let mut n_reps_numbers: Vec<u32> = Vec::with_capacity(n_rows);
    let mut accepted_numbers: Vec<u32> = Vec::with_capacity(n_rows);
    let mut sum_columns: Vec<String> = Vec::with_capacity(n_rows);
    let mut district_cols: Vec<Vec<Option<f64>>> = district_ids
        .iter()
        .map(|_| Vec::with_capacity(n_rows))
        .collect();

    for row in tallies {
        let n_d = row.n_districts as usize;
        for (k, key) in key_list.iter().enumerate() {
            sample_numbers.push(row.sample_num);
            n_reps_numbers.push(row.n_reps);
            accepted_numbers.push(row.accepted_count);
            sum_columns.push(key.clone());

            let offset = k * n_d;
            for (ci, &d) in district_ids.iter().enumerate() {
                let di = d as usize;
                let present = di < n_d && (row.observed & (1u128 << d)) != 0;
                district_cols[ci].push(if present { Some(row.totals[offset + di]) } else { None });
            }
        }
    }

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_numbers).into(),
        Series::new("n_reps".into(), n_reps_numbers).into(),
        Series::new("accepted_count".into(), accepted_numbers).into(),
        Series::new("sum_columns".into(), sum_columns).into(),
    ])?;

    for (ci, &d) in district_ids.iter().enumerate() {
        let col = std::mem::take(&mut district_cols[ci]);
        df.with_column(Series::new(format!("district_{}", d).into(), col).into())?;
    }

    let mut file = File::create(file_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(None))
        .finish(&mut df)?;

    Ok(())
}

pub fn tally_and_save_from_key_list(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    key_list: Vec<String>,
    show_progress: bool,
) -> io::Result<()> {
    let attr_col_indices: Vec<usize> = key_list
        .iter()
        .map(|k| {
            *graph
                .attr_index
                .get(k)
                .unwrap_or_else(|| panic!("key {:?} not pre-loaded on graph", k))
        })
        .collect();

    let n_pb_tics = 1000;
    let mut pb = if show_progress {
        Some(ProgressBar::new(n_pb_tics))
    } else {
        None
    };
    let mut pb_tics = 0;

    let mut ben_file = File::open(in_file_name).expect("BEN file not found");
    let line_checker = BenDecoder::new(&ben_file).expect("Failed to initialize decoder");

    let basename = Path::new(in_file_name)
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy();
    eprintln!("Reading {:?}...", basename);

    let mut line_count: usize = 0;
    for _ in line_checker.enumerate() {
        line_count += 1;
    }
    println!("Found {:?} unique plans in {:?}\r", line_count, basename);

    let pb_step_size = (line_count / n_pb_tics as usize) as u32;
    let mut previous_step = 0;

    ben_file.seek(SeekFrom::Start(0))?;
    let ben_reader = BufReader::new(ben_file);
    let decoder = BenDecoder::new(ben_reader).unwrap();

    let mut all_tallies: Vec<TallyRow> = Vec::with_capacity(line_count);

    let mut sample_count: u64 = 1;
    let mut accepted_count: u32 = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch: Vec<(Vec<u16>, u16)> = Vec::with_capacity(BATCH_SIZE);

    let start_time = Instant::now();
    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, n_reps)) => {
                batch.push((assignment, n_reps));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, n_reps)| {
                            let (totals, n_districts, observed) =
                                tally_keys(&graph, assignment, &attr_col_indices);
                            (*n_reps, totals, n_districts, observed)
                        })
                        .collect();

                    for (n_reps, totals, n_districts, observed) in results {
                        all_tallies.push(TallyRow {
                            sample_num: sample_count,
                            n_reps: n_reps as u32,
                            accepted_count,
                            totals,
                            n_districts,
                            observed,
                        });
                        sample_count += n_reps as u64;
                        accepted_count += 1;
                    }
                    batch.clear();
                }
            }
            Err(e) => panic!("Error: {:?}", e),
        }
        if show_progress && accepted_count - previous_step >= pb_step_size {
            let pb_ref = pb.as_mut().unwrap();
            pb_ref.inc();

            let elapsed_secs = start_time.elapsed().as_secs_f64();
            let rate = (pb_tics + 1) as f64 / elapsed_secs;
            let remaining_secs = (n_pb_tics - pb_tics - 1) as f64 / rate;
            let elapsed_mins = (elapsed_secs / 60.0).floor() as u64;
            let elapsed_remain_secs = (elapsed_secs % 60.0) as u64;
            let remaining_mins = (remaining_secs / 60.0).floor() as u64;
            let remaining_remain_secs = (remaining_secs % 60.0) as u64;
            pb_ref.message(&format!(
                "Elapsed: {}m {}s, ETA: {}m {}s ",
                elapsed_mins, elapsed_remain_secs, remaining_mins, remaining_remain_secs
            ));
            pb_tics += 1;
            io::stderr().flush().unwrap();
            io::stdout().flush().unwrap();
            previous_step = accepted_count;
        }
    }

    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|(assignment, n_reps)| {
                let (totals, n_districts, observed) =
                    tally_keys(&graph, assignment, &attr_col_indices);
                (*n_reps, totals, n_districts, observed)
            })
            .collect();
        for (n_reps, totals, n_districts, observed) in results {
            all_tallies.push(TallyRow {
                sample_num: sample_count,
                n_reps: n_reps as u32,
                accepted_count,
                totals,
                n_districts,
                observed,
            });
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    if let Some(pb_ref) = pb.as_mut() {
        pb_ref.finish();
    }

    eprintln!("Writing final output...");
    save_tallies_to_parquet(out_file_name, &all_tallies, &key_list)
        .expect("Unable to save tallies");
    eprintln!("Done!");
    Ok(())
}
