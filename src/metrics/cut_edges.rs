use crate::district::{observe_district, observed_assignment_districts, MAX_DISTRICTS};
use crate::graph::Graph;
use crate::input::BenSource;
use crate::metrics::twodelta::{
    run_incremental_twodelta, DeltaChange, IncrementalTwoDeltaMetric, PostDeltaLabels, TwoDeltaRow,
    TwoDeltaRunOptions,
};
use crate::output::parquet::F64MetricWriter;
use crate::pipeline::{
    parquet_compression, run_pipeline, AssignmentLengthCheck, PARQUET_BATCH_ROWS,
};
use ben::BenVariant;
use std::fs::File;

/// Count cut edges for a single assignment, and capture the district label set.
///
/// `graph.edges` is a flat `Vec<(u32, u32)>` and — when the caller asked for a weighted tally —
/// `graph.edge_weights` is a parallel `Vec<f64>` resolved once at load time. The hot loop is a
/// straight pass over both, with no hashing and no per-sample string lookups.
///
/// Returns `(cut_value, observed_districts)`. The observed mask is taken from the assignment
/// itself, not the edge endpoints: an isolated node (degree 0) never appears as an endpoint, so an
/// edge-derived set would let such a node change districts without tripping the pipeline's
/// fixed-district-set check. The assignment pass is also cheaper — one `observe_district` per node
/// instead of two per edge.
fn cut_edges(graph: &Graph, assignment: &[u16]) -> crate::error::Result<(f64, u128)> {
    let (_n_districts, observed) = observed_assignment_districts(assignment)?;
    let cut_value = match &graph.edge_weights {
        Some(weights) => {
            let mut total = 0.0f64;
            for (edge_index, &(node_u, node_v)) in graph.edges.iter().enumerate() {
                if assignment[node_u as usize] != assignment[node_v as usize] {
                    total += weights[edge_index];
                }
            }
            total
        }
        None => {
            let mut count: u64 = 0;
            for &(node_u, node_v) in graph.edges.iter() {
                if assignment[node_u as usize] != assignment[node_v as usize] {
                    count += 1;
                }
            }
            count as f64
        }
    };
    Ok((cut_value, observed))
}

/// Maintains cut-edge totals across TwoDelta events without rescanning every edge.
///
/// `update_delta` expects its `before` slice to still be the pre-delta assignment; the caller
/// applies the changes only after the cut-edge total has been patched.
struct IncrementalCutEdges<'g> {
    graph: &'g Graph,
    cut_value: f64,
    node_counts: Vec<u32>,
    observed: u128,
    post_delta_labels: PostDeltaLabels,
    seen_edges: Vec<u64>,
    gen: u64,
}

impl<'g> IncrementalCutEdges<'g> {
    fn new(graph: &'g Graph) -> Self {
        Self {
            graph,
            cut_value: 0.0,
            node_counts: vec![0; MAX_DISTRICTS as usize],
            observed: 0,
            post_delta_labels: PostDeltaLabels::new(graph.node_count),
            seen_edges: vec![0; graph.edges.len()],
            gen: 0,
        }
    }

    /// Recompute all state from a snapshot assignment.
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        let (cut_value, observed) = cut_edges(self.graph, assignment)?;
        self.cut_value = cut_value;
        self.observed = observed;
        self.node_counts.fill(0);
        for &district in assignment {
            self.node_counts[district as usize] += 1;
        }
        Ok(())
    }

    /// Apply one delta event to the maintained cut-edge total and district set.
    ///
    /// This evaluates each incident edge once, then leaves assignment mutation to the caller so
    /// edges with both endpoints changed see a consistent pre/post comparison.
    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[DeltaChange],
    ) -> crate::error::Result<()> {
        for change in changes {
            observe_district(&mut self.observed, change.new)?;
            self.node_counts[change.new as usize] += 1;
            self.node_counts[change.old as usize] -= 1;
            if self.node_counts[change.old as usize] == 0 {
                self.observed &= !(1u128 << change.old);
            }
        }

        self.post_delta_labels.refresh(changes);
        self.gen += 1;
        for change in changes {
            for &(_neighbor, edge_index) in self.graph.neighbors(change.node) {
                let edge_index = edge_index as usize;
                if self.seen_edges[edge_index] == self.gen {
                    continue;
                }
                self.seen_edges[edge_index] = self.gen;
                let (u, v) = self.graph.edges[edge_index];
                let u = u as usize;
                let v = v as usize;
                let was_cut = before[u] != before[v];
                let is_cut = self.post_delta_labels.label(before, u)
                    != self.post_delta_labels.label(before, v);
                if was_cut != is_cut {
                    let weight = self
                        .graph
                        .edge_weights
                        .as_ref()
                        .map_or(1.0, |weights| weights[edge_index]);
                    self.cut_value += if is_cut { weight } else { -weight };
                }
            }
        }

        Ok(())
    }
}

impl IncrementalTwoDeltaMetric for IncrementalCutEdges<'_> {
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        IncrementalCutEdges::seed(self, assignment)
    }

    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[DeltaChange],
    ) -> crate::error::Result<()> {
        IncrementalCutEdges::update_delta(self, before, changes)
    }

    fn observed(&self) -> u128 {
        self.observed
    }
}

pub fn tally_and_save_cut_edges(
    graph: Graph,
    source: &BenSource,
    out_file_name: &str,
    show_progress: bool,
    max_samples: Option<usize>,
    high_compression: bool,
) -> crate::error::Result<()> {
    // The output file is created lazily on the first decoded assignment (or at finish for a
    // zero-frame run), so a run that fails before producing data leaves nothing on disk.
    let out_path = out_file_name.to_string();
    let mut writer = F64MetricWriter::new(
        Box::new(move || File::create(out_path)),
        "cut_edges",
        parquet_compression(high_compression),
        PARQUET_BATCH_ROWS,
    );

    if source.variant()? == BenVariant::TwoDelta && graph.adjacency.is_some() {
        let mut state = IncrementalCutEdges::new(&graph);
        run_incremental_twodelta(
            source,
            TwoDeltaRunOptions {
                expected_len: graph.node_count,
                expected_len_label: "graph node count",
                output_name: "cut-edges",
                show_progress,
                max_samples,
            },
            &mut state,
            |state,
             TwoDeltaRow {
                 step,
                 n_reps,
                 accepted,
             }| { writer.push_row(step, n_reps, accepted, state.cut_value) },
        )?;
    } else {
        run_pipeline(
            source,
            AssignmentLengthCheck::Exact {
                expected: graph.node_count,
                label: "graph node count",
            },
            "cut-edges",
            |assignment, _n_reps| {
                let (cuts, observed) = cut_edges(&graph, assignment)?;
                Ok((observed, cuts))
            },
            |step, n_reps, accepted, cuts| writer.push_row(step, n_reps, accepted, cuts),
            show_progress,
            max_samples,
        )?;
    }

    log::info!("Writing final output...");
    writer.finish()?;

    log::info!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cut_edges, DeltaChange, IncrementalCutEdges};
    use crate::graph::{CsrAdjacency, Graph};
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
            adjacency: None,
        }
    }

    #[test]
    fn cut_edges_counts_unweighted_crossings() {
        let graph = graph_with_edges(None);
        // Each call also returns the district set folded in from the edge endpoints it walked.
        let d12 = (1u128 << 1) | (1u128 << 2);
        assert_eq!(cut_edges(&graph, &[1, 1, 2, 2]).unwrap(), (1.0, d12));
        assert_eq!(cut_edges(&graph, &[1, 2, 1, 2]).unwrap(), (3.0, d12));
        assert_eq!(cut_edges(&graph, &[7, 7, 7, 7]).unwrap(), (0.0, 1u128 << 7));
    }

    #[test]
    fn cut_edges_sums_aligned_weights_for_crossings() {
        let graph = graph_with_edges(Some(vec![2.0, 5.5, 3.0]));
        let d12 = (1u128 << 1) | (1u128 << 2);
        assert_eq!(cut_edges(&graph, &[1, 1, 2, 2]).unwrap(), (5.5, d12));
        assert_eq!(cut_edges(&graph, &[1, 2, 1, 2]).unwrap(), (10.5, d12));
        assert_eq!(cut_edges(&graph, &[4, 4, 4, 4]).unwrap(), (0.0, 1u128 << 4));
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
            adjacency: None,
        }
    }

    fn graph_with_path_adjacency() -> Graph {
        Graph {
            node_count: 4,
            attr_columns: vec![],
            attr_index: HashMap::new(),
            region_columns: vec![],
            region_index: HashMap::new(),
            region_id_counts: vec![],
            edges: vec![(0, 1), (1, 2), (2, 3)],
            edge_weights: None,
            adjacency: Some(CsrAdjacency {
                offsets: vec![0, 1, 3, 5, 6],
                neighbors: vec![(1, 0), (0, 0), (2, 1), (1, 1), (3, 2), (2, 2)],
            }),
        }
    }

    #[test]
    fn cut_edges_returns_zero_for_empty_edge_list() {
        // No edges → no crossings, regardless of assignment. Both the weighted and unweighted code
        // paths must return 0.0 cleanly (no panic on the zip with weights, no negative-length iter
        // issues). The district set is captured from the assignment, not the edge endpoints, so it
        // is full even with no edges — pinning that the cut-edges set is node-derived.
        let d123 = (1u128 << 1) | (1u128 << 2) | (1u128 << 3);
        let unweighted = graph_with_explicit_edges(vec![], None);
        let weighted = graph_with_explicit_edges(vec![], Some(vec![]));
        assert_eq!(cut_edges(&unweighted, &[1, 2, 3]).unwrap(), (0.0, d123));
        assert_eq!(cut_edges(&weighted, &[1, 2, 3]).unwrap(), (0.0, d123));
    }

    #[test]
    fn incremental_cut_edges_handles_edge_with_both_endpoints_moved() {
        let graph = graph_with_path_adjacency();
        let before = vec![1, 1, 2, 2];
        let changes = vec![
            DeltaChange {
                node: 1,
                old: 1,
                new: 2,
            },
            DeltaChange {
                node: 2,
                old: 2,
                new: 1,
            },
        ];
        let mut state = IncrementalCutEdges::new(&graph);

        state.seed(&before).unwrap();
        state.update_delta(&before, &changes).unwrap();

        assert_eq!(state.cut_value, 3.0);
        assert_eq!(state.observed, (1u128 << 1) | (1u128 << 2));
    }

    #[test]
    fn cut_edges_observes_isolated_node_districts() {
        // Node 2 has no incident edges. Its district must still appear in the observed set, or the
        // pipeline's fixed-district-set guard could never see it change across frames.
        let graph = Graph {
            node_count: 3,
            attr_columns: vec![],
            attr_index: HashMap::new(),
            region_columns: vec![],
            region_index: HashMap::new(),
            region_id_counts: vec![],
            edges: vec![(0, 1)],
            edge_weights: None,
            adjacency: None,
        };
        let d125 = (1u128 << 1) | (1u128 << 2) | (1u128 << 5);
        assert_eq!(cut_edges(&graph, &[1, 2, 5]).unwrap(), (1.0, d125));
    }

    #[test]
    fn cut_edges_errors_on_district_beyond_limit() {
        // A district id >= 128 can't fit the u128 observed bitmask. cut_edges captures the set
        // from the assignment, so it must surface a clean error rather than silently wrapping the
        // shift.
        let graph = graph_with_edges(None);
        let err = cut_edges(&graph, &[1, 1, 2, 128]).unwrap_err();
        assert!(
            err.to_string()
                .contains("exceeds current 128-district limit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn cut_edges_handles_zero_and_negative_weights() {
        // The weighted path is a straight sum; there's no clamp on the inputs. Zero-weight cut
        // edges contribute 0; negative weights subtract. Pin the behavior so a future "saturate at
        // zero" change would be caught.
        let graph = graph_with_edges(Some(vec![0.0, -2.5, 4.0]));
        let d12 = (1u128 << 1) | (1u128 << 2);
        // Assignment [1,2,1,2] cuts every edge: 0.0 + (-2.5) + 4.0 = 1.5.
        assert_eq!(cut_edges(&graph, &[1, 2, 1, 2]).unwrap(), (1.5, d12));
        // Assignment [1,1,2,2] cuts only edge (1,2) which has weight -2.5.
        assert_eq!(cut_edges(&graph, &[1, 1, 2, 2]).unwrap(), (-2.5, d12));
    }

    #[test]
    fn cut_edges_batched_writer_appends_multiple_batches() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let mut writer = F64MetricWriter::new(
            Box::new(move || File::create(path)),
            "cut_edges",
            parquet_compression(false),
            2,
        );

        writer.push_row(1, 1, 1, 3.0).unwrap();
        writer.push_row(2, 1, 2, 4.0).unwrap();
        writer.push_row(3, 2, 3, 9.5).unwrap();
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
