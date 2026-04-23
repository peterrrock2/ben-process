use crate::graph::Graph;
use ben::decode::{count_samples_from_file, BenDecoder};
use pbr::ProgressBar;
use polars::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;

/// Count cut edges for a single assignment.
///
/// `graph.edges` is a flat `Vec<(u32, u32)>` and — when the caller asked for
/// a weighted tally — `graph.edge_weights` is a parallel `Vec<f64>` resolved
/// once at load time. The hot loop is a straight pass over both, with no
/// hashing and no per-sample string lookups.
fn cut_edges(graph: &Graph, assignment: &[u16]) -> f64 {
    match &graph.edge_weights {
        Some(weights) => {
            let mut total = 0.0f64;
            for (i, &(u, v)) in graph.edges.iter().enumerate() {
                if assignment[u as usize] != assignment[v as usize] {
                    total += weights[i];
                }
            }
            total
        }
        None => {
            let mut count: u64 = 0;
            for &(u, v) in graph.edges.iter() {
                if assignment[u as usize] != assignment[v as usize] {
                    count += 1;
                }
            }
            count as f64
        }
    }
}

pub fn tally_and_save_cut_edges(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
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

    let mut sample_nums = Vec::with_capacity(line_count);
    let mut n_reps_nums = Vec::with_capacity(line_count);
    let mut accepted_nums = Vec::with_capacity(line_count);
    let mut cut_edge_counts = Vec::with_capacity(line_count);

    let mut sample_count: u64 = 1;
    let mut accepted_count: u32 = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch: Vec<(Vec<u16>, u16)> = Vec::with_capacity(BATCH_SIZE);

    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, count)) => {
                batch.push((assignment, count));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, count)| (*count, cut_edges(&graph, assignment)))
                        .collect();

                    for (n_reps, counts) in results {
                        sample_nums.push(sample_count);
                        n_reps_nums.push(n_reps as u32);
                        accepted_nums.push(accepted_count);
                        cut_edge_counts.push(counts);
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
            .map(|(assignment, count)| (*count, cut_edges(&graph, assignment)))
            .collect();
        for (n_reps, counts) in results {
            sample_nums.push(sample_count);
            n_reps_nums.push(n_reps as u32);
            accepted_nums.push(accepted_count);
            cut_edge_counts.push(counts);
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    if let Some(pb_ref) = pb.as_mut() {
        pb_ref.finish();
    }

    println!();

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_nums).into(),
        Series::new("n_reps".into(), n_reps_nums).into(),
        Series::new("accepted_count".into(), accepted_nums).into(),
        Series::new("cut_edges".into(), cut_edge_counts).into(),
    ])?;

    let mut file = File::create(out_file_name)?;

    eprintln!("Writing final output...");
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(None))
        .finish(&mut df)?;

    eprintln!("Done!");
    Ok(())
}
