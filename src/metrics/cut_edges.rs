use crate::graph::Graph;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
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

fn cut_edges_batch_to_df(
    sample_nums: &mut Vec<u64>,
    n_reps_nums: &mut Vec<u32>,
    accepted_nums: &mut Vec<u32>,
    cut_edge_counts: &mut Vec<f64>,
) -> PolarsResult<DataFrame> {
    DataFrame::new_infer_height(vec![
        Series::new("step".into(), std::mem::take(sample_nums)).into(),
        Series::new("n_reps".into(), std::mem::take(n_reps_nums)).into(),
        Series::new("accepted_count".into(), std::mem::take(accepted_nums)).into(),
        Series::new("cut_edges".into(), std::mem::take(cut_edge_counts)).into(),
    ])
}

pub fn tally_and_save_cut_edges(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    show_progress: bool,
    high_compression: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let total = count_samples(in_file_name)?;

    let mut file = File::create(out_file_name)?;
    let empty_df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), Vec::<u64>::new()).into(),
        Series::new("n_reps".into(), Vec::<u32>::new()).into(),
        Series::new("accepted_count".into(), Vec::<u32>::new()).into(),
        Series::new("cut_edges".into(), Vec::<f64>::new()).into(),
    ])?;
    let mut writer = ParquetWriter::new(&mut file)
        .with_compression(parquet_compression(high_compression))
        .batched(empty_df.schema())?;

    let mut sample_nums: Vec<u64> = Vec::with_capacity(PARQUET_BATCH_ROWS);
    let mut n_reps_nums: Vec<u32> = Vec::with_capacity(PARQUET_BATCH_ROWS);
    let mut accepted_nums: Vec<u32> = Vec::with_capacity(PARQUET_BATCH_ROWS);
    let mut cut_edge_counts: Vec<f64> = Vec::with_capacity(PARQUET_BATCH_ROWS);

    run_pipeline(
        in_file_name,
        total,
        |assignment, _n_reps| cut_edges(&graph, assignment),
        |step, n_reps, accepted, cuts| {
            sample_nums.push(step);
            n_reps_nums.push(n_reps);
            accepted_nums.push(accepted);
            cut_edge_counts.push(cuts);
            if sample_nums.len() >= PARQUET_BATCH_ROWS {
                let df = cut_edges_batch_to_df(
                    &mut sample_nums,
                    &mut n_reps_nums,
                    &mut accepted_nums,
                    &mut cut_edge_counts,
                )
                .expect("Unable to build cut-edges batch DataFrame");
                writer
                    .write_batch(&df)
                    .expect("Unable to write cut-edges batch");
            }
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    if !sample_nums.is_empty() {
        let df = cut_edges_batch_to_df(
            &mut sample_nums,
            &mut n_reps_nums,
            &mut accepted_nums,
            &mut cut_edge_counts,
        )?;
        writer.write_batch(&df)?;
    }
    writer.finish()?;

    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cut_edges, cut_edges_batch_to_df};
    use crate::graph::Graph;
    use crate::pipeline::parquet_compression;
    use polars::prelude::{ParquetReader, ParquetWriter, SerReader};
    use std::collections::HashMap;
    use std::fs::File;
    use tempfile::NamedTempFile;

    fn graph_with_edges(edge_weights: Option<Vec<f64>>) -> Graph {
        Graph {
            attr_columns: vec![],
            attr_index: HashMap::new(),
            region_columns: vec![],
            region_index: HashMap::new(),
            region_id_counts: vec![],
            edges: vec![(0, 1), (1, 2), (2, 3)],
            edge_weights,
        }
    }

    #[test]
    fn cut_edges_counts_unweighted_crossings() {
        let graph = graph_with_edges(None);
        assert_eq!(cut_edges(&graph, &[1, 1, 2, 2]), 1.0);
        assert_eq!(cut_edges(&graph, &[1, 2, 1, 2]), 3.0);
        assert_eq!(cut_edges(&graph, &[7, 7, 7, 7]), 0.0);
    }

    #[test]
    fn cut_edges_sums_aligned_weights_for_crossings() {
        let graph = graph_with_edges(Some(vec![2.0, 5.5, 3.0]));
        assert_eq!(cut_edges(&graph, &[1, 1, 2, 2]), 5.5);
        assert_eq!(cut_edges(&graph, &[1, 2, 1, 2]), 10.5);
        assert_eq!(cut_edges(&graph, &[4, 4, 4, 4]), 0.0);
    }

    #[test]
    fn cut_edges_batched_writer_appends_multiple_batches() {
        let file = NamedTempFile::new().unwrap();
        let mut empty_steps = vec![];
        let mut empty_reps = vec![];
        let mut empty_accepted = vec![];
        let mut empty_counts = vec![];
        let empty_df = cut_edges_batch_to_df(
            &mut empty_steps,
            &mut empty_reps,
            &mut empty_accepted,
            &mut empty_counts,
        )
        .unwrap();
        let mut writer = ParquetWriter::new(File::create(file.path()).unwrap())
            .with_compression(parquet_compression(false))
            .batched(empty_df.schema())
            .unwrap();

        let mut batch1_steps = vec![1, 2];
        let mut batch1_reps = vec![1, 1];
        let mut batch1_accepted = vec![1, 2];
        let mut batch1_counts = vec![3.0, 4.0];
        let batch1 = cut_edges_batch_to_df(
            &mut batch1_steps,
            &mut batch1_reps,
            &mut batch1_accepted,
            &mut batch1_counts,
        )
        .unwrap();
        writer.write_batch(&batch1).unwrap();

        let mut batch2_steps = vec![3];
        let mut batch2_reps = vec![2];
        let mut batch2_accepted = vec![3];
        let mut batch2_counts = vec![9.5];
        let batch2 = cut_edges_batch_to_df(
            &mut batch2_steps,
            &mut batch2_reps,
            &mut batch2_accepted,
            &mut batch2_counts,
        )
        .unwrap();
        writer.write_batch(&batch2).unwrap();
        writer.finish().unwrap();

        let df = ParquetReader::new(&mut File::open(file.path()).unwrap())
            .finish()
            .unwrap();
        assert_eq!(
            df.column("step")
                .unwrap()
                .u64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            df.column("cut_edges")
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![3.0, 4.0, 9.5]
        );
    }
}
