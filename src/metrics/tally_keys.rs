use crate::graph::Graph;
use ben::decode::BenDecoder;
use pbr::ProgressBar;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

/// Sum each requested attribute column into per-district buckets.
///
/// Output preserves the old shape (`HashMap<String, HashMap<u16, f64>>`) so
/// `save_tallies_to_parquet` is unchanged. The speedup vs. the previous
/// implementation comes from sourcing values from `graph.attr_columns`
/// (pre-parsed `f64`) instead of re-walking `serde_json::Value` on every call.
fn tally_keys(
    graph: &Graph,
    assignment: &[u16],
    key_list: &[String],
    attr_col_indices: &[usize],
) -> HashMap<String, HashMap<u16, f64>> {
    let partition_values: HashSet<u16> = assignment.iter().copied().collect();

    let mut tallies: HashMap<String, HashMap<u16, f64>> = key_list
        .iter()
        .map(|k| (k.clone(), partition_values.iter().map(|&d| (d, 0.0)).collect()))
        .collect();

    for (k_idx, key) in key_list.iter().enumerate() {
        let col = &graph.attr_columns[attr_col_indices[k_idx]];
        let sub = tallies.get_mut(key).unwrap();
        for (i, &v) in col.iter().enumerate() {
            *sub.get_mut(&assignment[i]).unwrap() += v;
        }
    }

    tallies
}

/// Given a (parquet) file path to save to and a list of tallies corresponding to values obtained from
/// assessing some function over an ensemble of graph partitions, this function saves the tallies
/// to a Parquet file.
fn save_tallies_to_parquet(
    file_path: &str,
    tallies: &Vec<(u64, u32, u32, HashMap<String, HashMap<u16, f64>>)>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut sample_numbers = Vec::new();
    let mut n_reps_numbers = Vec::new();
    let mut accepted_numbers = Vec::new();

    let mut keys = Vec::new();
    // BTreeMap so district columns come out in a stable, sorted order.
    let mut partition_data: BTreeMap<u16, Vec<Option<f64>>> = BTreeMap::new();

    // Initialize partition_data with empty vectors for each unique partition key
    for (_, _, _, tally) in tallies {
        for (_, sub_map) in tally {
            for (&partition_key, _) in sub_map {
                partition_data.entry(partition_key).or_insert_with(Vec::new);
            }
        }
    }

    // Fill in the data
    for (sample_num, n_reps, accepted_num, tally) in tallies {
        for (key, sub_map) in tally {
            sample_numbers.push(*sample_num);
            n_reps_numbers.push(*n_reps);
            accepted_numbers.push(*accepted_num);
            keys.push(key.clone());
            for (&partition_key, value) in partition_data.iter_mut() {
                value.push(sub_map.get(&partition_key).copied());
            }
        }
    }

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_numbers).into(),
        Series::new("n_reps".into(), n_reps_numbers).into(),
        Series::new("accepted_count".into(), accepted_numbers).into(),
        Series::new("sum_columns".into(), keys).into(),
    ])?;

    for (partition_key, values) in partition_data {
        df.with_column(Series::new(format!("district_{}", partition_key).into(), values).into())?;
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
    // Resolve requested keys to column indices once, up-front. load_graph
    // already parsed these columns; if any key is missing we want to fail
    // loudly here rather than silently per-sample.
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

    let mut all_tallies: Vec<(u64, u32, u32, HashMap<String, HashMap<u16, f64>>)> =
        Vec::with_capacity(line_count);

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
                            (
                                *n_reps,
                                tally_keys(&graph, assignment, &key_list, &attr_col_indices),
                            )
                        })
                        .collect();

                    for (n_reps, tallies) in results {
                        all_tallies.push((sample_count, n_reps as u32, accepted_count, tallies));
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
                (
                    *n_reps,
                    tally_keys(&graph, assignment, &key_list, &attr_col_indices),
                )
            })
            .collect();
        for (n_reps, tallies) in results {
            all_tallies.push((sample_count, n_reps as u32, accepted_count, tallies));
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    if let Some(pb_ref) = pb.as_mut() {
        pb_ref.finish();
    }

    eprintln!("Writing final output...");
    save_tallies_to_parquet(out_file_name, &all_tallies).expect("Unable to save tallies");
    eprintln!("Done!");
    Ok(())
}
