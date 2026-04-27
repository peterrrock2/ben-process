use crate::graph::Graph;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline};
use polars::prelude::*;
use std::fs::File;

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
    high_compression: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let total = count_samples(in_file_name)?;

    let mut sample_nums: Vec<u64> = Vec::with_capacity(total);
    let mut n_reps_nums: Vec<u32> = Vec::with_capacity(total);
    let mut accepted_nums: Vec<u32> = Vec::with_capacity(total);
    let mut cut_edge_counts: Vec<f64> = Vec::with_capacity(total);

    run_pipeline(
        in_file_name,
        total,
        |assignment, _n_reps| cut_edges(&graph, assignment),
        |step, n_reps, accepted, cuts| {
            sample_nums.push(step);
            n_reps_nums.push(n_reps);
            accepted_nums.push(accepted);
            cut_edge_counts.push(cuts);
        },
        show_progress,
    )?;

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_nums).into(),
        Series::new("n_reps".into(), n_reps_nums).into(),
        Series::new("accepted_count".into(), accepted_nums).into(),
        Series::new("cut_edges".into(), cut_edge_counts).into(),
    ])?;

    let mut file = File::create(out_file_name)?;

    eprintln!("Writing final output...");
    ParquetWriter::new(&mut file)
        .with_compression(parquet_compression(high_compression))
        .finish(&mut df)?;

    eprintln!("Done!");
    Ok(())
}
