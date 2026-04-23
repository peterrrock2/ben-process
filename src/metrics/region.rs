use crate::graph::Graph;
use ben::decode::{count_samples_from_file, BenDecoder};
use pbr::ProgressBar;
use polars::prelude::*;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

#[derive(Clone, Copy)]
pub enum RegionMetric {
    Splits,
    Pieces,
}

fn parse_region_id(node: &Value, key: &str) -> Option<String> {
    let value = &node[key];
    match value {
        Value::Null => None,
        Value::Number(n) => {
            let v = n.as_f64()?;
            if v.is_nan() {
                None
            } else {
                Some(value.to_string())
            }
        }
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("nan") {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        Value::Bool(_) => Some(value.to_string()),
        _ => None,
    }
}

fn region_metric_for_key(
    graph: &Graph,
    assignment: &Vec<u16>,
    key: &str,
    metric: RegionMetric,
) -> u32 {
    let mut region_to_districts: HashMap<String, HashSet<u16>> = HashMap::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        if let Some(region_id) = parse_region_id(node, key) {
            region_to_districts
                .entry(region_id)
                .or_insert_with(HashSet::new)
                .insert(assignment[idx]);
        }
    }

    match metric {
        RegionMetric::Splits => region_to_districts
            .values()
            .filter(|districts| districts.len() > 1)
            .count() as u32,
        RegionMetric::Pieces => region_to_districts
            .values()
            .map(|districts| districts.len() as u32)
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

    let mut sample_count = 1;
    let mut accepted_count = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, n_reps)) => {
                batch.push((assignment, n_reps));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, n_reps)| {
                            let counts = key_list
                                .iter()
                                .map(|key| {
                                    (
                                        key.clone(),
                                        region_metric_for_key(&graph, assignment, key, metric),
                                    )
                                })
                                .collect::<Vec<(String, u32)>>();
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
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        }
    }

    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|(assignment, n_reps)| {
                let counts = key_list
                    .iter()
                    .map(|key| {
                        (
                            key.clone(),
                            region_metric_for_key(&graph, assignment, key, metric),
                        )
                    })
                    .collect::<Vec<(String, u32)>>();
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

    let mut file = File::create(out_file_name).expect(
        format!(
            "Failed to create output file {:?}. The file may already exist.",
            out_file_name
        )
        .as_str(),
    );
    eprintln!("Writing final output...");
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(None))
        .finish(&mut df)?;

    eprintln!("Done!");
    Ok(())
}
