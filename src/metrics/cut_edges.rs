use crate::graph::Graph;
use crate::output::parquet::F64MetricWriter;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use std::fs::File;

/// Count cut edges for a single assignment.
///
/// `graph.edges` is a flat `Vec<(u32, u32)>` and — when the caller asked for a weighted tally —
/// `graph.edge_weights` is a parallel `Vec<f64>` resolved once at load time. The hot loop is a
/// straight pass over both, with no hashing and no per-sample string lookups.
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

    let file = File::create(out_file_name)?;
    let mut writer = F64MetricWriter::new(
        file,
        "cut_edges",
        parquet_compression(high_compression),
        PARQUET_BATCH_ROWS,
    )?;

    run_pipeline(
        in_file_name,
        total,
        Some(graph.node_count),
        |assignment, _n_reps| cut_edges(&graph, assignment),
        |step, n_reps, accepted, cuts| {
            writer
                .push(step, n_reps, accepted, cuts)
                .expect("Unable to write cut-edges batch");
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    writer.finish()?;

    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::cut_edges;
    use crate::graph::Graph;
    use crate::output::parquet::F64MetricWriter;
    use crate::pipeline::parquet_compression;
    use polars::prelude::{ParquetReader, SerReader};
    use std::collections::HashMap;
    use std::fs::File;
    use tempfile::NamedTempFile;

    fn graph_with_edges(edge_weights: Option<Vec<f64>>) -> Graph {
        Graph {
            node_count: 4,
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

    fn graph_with_explicit_edges(edges: Vec<(u32, u32)>, edge_weights: Option<Vec<f64>>) -> Graph {
        let node_count = edges
            .iter()
            .map(|&(a, b)| a.max(b))
            .max()
            .map_or(0, |m| m as usize + 1);
        Graph {
            node_count,
            attr_columns: vec![],
            attr_index: HashMap::new(),
            region_columns: vec![],
            region_index: HashMap::new(),
            region_id_counts: vec![],
            edges,
            edge_weights,
        }
    }

    #[test]
    fn cut_edges_returns_zero_for_empty_edge_list() {
        // No edges → no crossings, regardless of assignment. Both the weighted and unweighted code
        // paths must return 0.0 cleanly (no panic on the zip with weights, no negative-length iter
        // issues).
        let unweighted = graph_with_explicit_edges(vec![], None);
        let weighted = graph_with_explicit_edges(vec![], Some(vec![]));
        assert_eq!(cut_edges(&unweighted, &[1, 2, 3]), 0.0);
        assert_eq!(cut_edges(&weighted, &[1, 2, 3]), 0.0);
    }

    #[test]
    fn cut_edges_handles_zero_and_negative_weights() {
        // The weighted path is a straight sum; there's no clamp on the inputs. Zero-weight cut
        // edges contribute 0; negative weights subtract. Pin the behavior so a future "saturate at
        // zero" change would be caught.
        let graph = graph_with_edges(Some(vec![0.0, -2.5, 4.0]));
        // Assignment [1,2,1,2] cuts every edge: 0.0 + (-2.5) + 4.0 = 1.5.
        assert_eq!(cut_edges(&graph, &[1, 2, 1, 2]), 1.5);
        // Assignment [1,1,2,2] cuts only edge (1,2) which has weight -2.5.
        assert_eq!(cut_edges(&graph, &[1, 1, 2, 2]), -2.5);
    }

    #[test]
    fn cut_edges_batched_writer_appends_multiple_batches() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = F64MetricWriter::new(
            File::create(file.path()).unwrap(),
            "cut_edges",
            parquet_compression(false),
            2,
        )
        .unwrap();

        writer.push(1, 1, 1, 3.0).unwrap();
        writer.push(2, 1, 2, 4.0).unwrap();
        writer.push(3, 2, 3, 9.5).unwrap();
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
