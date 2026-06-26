use crate::cli::{build_tally_output_dir, build_tally_output_path};
use crate::district::{
    observe_district, observed_assignment_districts, validate_district_set_unchanged, MAX_DISTRICTS,
};
use crate::graph::Graph;
use crate::input::BenSource;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{
    capped_reps, make_progress_bar, parquet_compression, run_pipeline, AssignmentLengthCheck,
    PARQUET_BATCH_ROWS,
};
use ben::io::reader::TwoDeltaFrameEvent;
use ben::BenVariant;
use std::fs::{create_dir_all, File};
use std::io;

/// Hot loop: flat index into pre-parsed attribute columns, accumulate into a flat per-district
/// totals vector. No HashMap work inside the inner loop.
///
/// `totals` is a flat `Vec<f64>` of shape `[n_keys * n_districts]`, where
/// `n_districts = max(assignment) + 1`. `observed` has bit `d` set iff district `d` appeared in
/// this sample's assignment.
fn tally_keys(
    graph: &Graph,
    assignment: &[u16],
    attr_column_indices: &[usize],
) -> crate::error::Result<(Vec<f64>, u16, u128)> {
    // The assignment is guaranteed to have one entry per graph node by `run_pipeline`'s length
    // check; this hot loop relies on that invariant when indexing `assignment[node_index]` below.
    let (n_districts, observed) = observed_assignment_districts(assignment)?;
    let n_districts = n_districts as usize;
    let n_keys = attr_column_indices.len();
    let mut totals = vec![0.0f64; n_keys * n_districts];
    for (key_index, &column_index) in attr_column_indices.iter().enumerate() {
        let column = &graph.attr_columns[column_index];
        let offset = key_index * n_districts;
        for (node_index, &value) in column.iter().enumerate() {
            totals[offset + assignment[node_index] as usize] += value;
        }
    }
    Ok((totals, n_districts as u16, observed))
}

/// Maintains per-key district totals across TwoDelta events.
///
/// `update_delta` expects `before` to still be the pre-delta assignment; the caller applies the
/// changes after the totals and district counts are patched.
struct IncrementalTallies<'g> {
    graph: &'g Graph,
    attr_column_indices: &'g [usize],
    totals: Vec<f64>,
    node_counts: Vec<u32>,
    observed: u128,
}

impl<'g> IncrementalTallies<'g> {
    fn new(graph: &'g Graph, attr_column_indices: &'g [usize]) -> Self {
        Self {
            graph,
            attr_column_indices,
            totals: vec![0.0; attr_column_indices.len() * MAX_DISTRICTS as usize],
            node_counts: vec![0; MAX_DISTRICTS as usize],
            observed: 0,
        }
    }

    /// Recompute all tallies and district counts from a snapshot assignment.
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        self.totals.fill(0.0);
        self.node_counts.fill(0);
        self.observed = 0;

        for (node, &district) in assignment.iter().enumerate() {
            observe_district(&mut self.observed, district)?;
            self.node_counts[district as usize] += 1;
            for (key_index, &column_index) in self.attr_column_indices.iter().enumerate() {
                let offset = key_index * MAX_DISTRICTS as usize;
                self.totals[offset + district as usize] +=
                    self.graph.attr_columns[column_index][node];
            }
        }

        Ok(())
    }

    /// Apply one delta event to the maintained tallies and district set.
    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[(usize, u16, u16)],
    ) -> crate::error::Result<()> {
        for &(node, old, new) in changes {
            let Some(&current) = before.get(node) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "TwoDelta delta references node {node} outside assignment length {}",
                        before.len()
                    ),
                )
                .into());
            };
            if current != old {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "TwoDelta delta old label mismatch at node {node}: \
                         expected {current}, got {old}",
                    ),
                )
                .into());
            }

            observe_district(&mut self.observed, old)?;
            observe_district(&mut self.observed, new)?;
            self.node_counts[new as usize] += 1;
            self.node_counts[old as usize] -= 1;
            if self.node_counts[old as usize] == 0 {
                self.observed &= !(1u128 << old);
            }

            for (key_index, &column_index) in self.attr_column_indices.iter().enumerate() {
                let value = self.graph.attr_columns[column_index][node];
                let offset = key_index * MAX_DISTRICTS as usize;
                self.totals[offset + old as usize] -= value;
                self.totals[offset + new as usize] += value;
            }
        }

        Ok(())
    }
}

fn push_tally_rows(
    writers: &mut [DistrictMetricWriter],
    step: u64,
    n_reps: u32,
    accepted: u64,
    observed: u128,
    totals: &[f64],
    n_districts: usize,
) -> crate::error::Result<()> {
    for (key_index, writer) in writers.iter_mut().enumerate() {
        let offset = key_index * n_districts;
        writer.push_row(
            step,
            n_reps,
            accepted,
            (observed, &totals[offset..offset + n_districts]),
        )?;
    }
    Ok(())
}

/// Run tally-keys directly from TwoDelta events, reseeding on snapshots and patching on deltas.
fn run_incremental_twodelta_tally_keys(
    graph: &Graph,
    source: &BenSource,
    writers: &mut [DistrictMetricWriter],
    attr_column_indices: &[usize],
    show_progress: bool,
    max_samples: Option<usize>,
) -> crate::error::Result<()> {
    let progress_bar = if show_progress {
        Some(make_progress_bar(match max_samples {
            Some(n) => n,
            None => source.count_samples()?,
        }))
    } else {
        None
    };

    let mut remaining_samples = max_samples;
    let mut assignment: Option<Vec<u16>> = None;
    let mut expected_observed: Option<u128> = None;
    let mut state = IncrementalTallies::new(graph, attr_column_indices);
    let mut step = 1u64;

    for (accepted, event) in (1u64..).zip(source.open_reader()?.into_twodelta_events()) {
        if remaining_samples == Some(0) {
            break;
        }

        let n_reps = match event? {
            TwoDeltaFrameEvent::Snapshot {
                assignment: snapshot,
                count,
                ..
            } => {
                if snapshot.len() != graph.node_count {
                    return Err(crate::error::BenError::AssignmentLength {
                        actual: snapshot.len(),
                        expected: graph.node_count,
                    });
                }
                state.seed(&snapshot)?;
                assignment = Some(snapshot);
                capped_reps(&mut remaining_samples, count)
            }
            TwoDeltaFrameEvent::Delta { changes, count } => {
                let assignment = assignment.as_mut().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "TwoDelta delta event appeared before an initial snapshot",
                    )
                })?;
                let changes = changes
                    .into_iter()
                    .map(|(node, old, new)| (node as usize, old, new))
                    .collect::<Vec<_>>();
                state.update_delta(assignment, &changes)?;
                for (node, _old, new) in changes {
                    assignment[node] = new;
                }
                capped_reps(&mut remaining_samples, count)
            }
        };

        match expected_observed {
            None => expected_observed = Some(state.observed),
            Some(expected) => validate_district_set_unchanged(state.observed, expected, "tally")?,
        }

        push_tally_rows(
            writers,
            step,
            n_reps as u32,
            accepted,
            state.observed,
            &state.totals,
            MAX_DISTRICTS as usize,
        )?;
        step += n_reps as u64;
        if let Some(progress_bar) = &progress_bar {
            progress_bar.inc(n_reps as u64);
        }
    }

    if let Some(progress_bar) = progress_bar {
        progress_bar.finish_and_clear();
    }
    Ok(())
}

pub fn tally_and_save_from_key_list(
    graph: Graph,
    source: &BenSource,
    output_dir: Option<&str>,
    key_list: Vec<String>,
    show_progress: bool,
    max_samples: Option<usize>,
    high_compression: bool,
) -> crate::error::Result<()> {
    let attr_column_indices: Vec<usize> = key_list
        .iter()
        .map(|key| {
            graph
                .numeric_column_index(key)
                .unwrap_or_else(|| panic!("key {:?} not pre-loaded on graph", key))
        })
        .collect();

    // One writer per key, each owning its output path. No file (and no tallies directory) is
    // created here: the writer defers that to the first decoded assignment, so a run that fails
    // before producing data leaves nothing on disk.
    // The original input path drives the per-key output names.
    let in_name = source.path().to_string_lossy();
    let mut writers: Vec<DistrictMetricWriter> = key_list
        .iter()
        .map(|key| {
            let tally_dir = build_tally_output_dir(&in_name, output_dir);
            let output_path = build_tally_output_path(&in_name, key, max_samples, output_dir);
            DistrictMetricWriter::new(
                Box::new(move || {
                    create_dir_all(&tally_dir)?;
                    File::create(output_path)
                }),
                parquet_compression(high_compression),
                PARQUET_BATCH_ROWS,
            )
        })
        .collect();

    if source.variant()? == BenVariant::TwoDelta {
        run_incremental_twodelta_tally_keys(
            &graph,
            source,
            &mut writers,
            &attr_column_indices,
            show_progress,
            max_samples,
        )?;
    } else {
        run_pipeline(
            source,
            AssignmentLengthCheck::MatchesGraph(graph.node_count),
            // The pipeline enforces that the district set is identical for every plan, so the
            // schema each writer fixes from its first row holds for the whole run.
            "tally",
            |assignment, _n_reps| {
                let (totals, n_districts, observed) =
                    tally_keys(&graph, assignment, &attr_column_indices)?;
                Ok((observed, (totals, n_districts, observed)))
            },
            |step, n_reps, accepted, (totals, n_districts, observed)| {
                push_tally_rows(
                    &mut writers,
                    step,
                    n_reps,
                    accepted,
                    observed,
                    &totals,
                    n_districts as usize,
                )
            },
            show_progress,
            max_samples,
        )?;
    }

    log::info!("Writing final output...");
    for writer in writers {
        writer.finish()?;
    }
    log::info!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{tally_keys, IncrementalTallies};
    use crate::graph::Graph;
    use std::collections::HashMap;

    fn graph_with_attr_columns(attr_columns: Vec<Vec<f64>>) -> Graph {
        Graph {
            node_count: attr_columns.first().map_or(0, |c| c.len()),
            attr_columns,
            attr_index: HashMap::new(),
            region_columns: vec![],
            region_index: HashMap::new(),
            region_id_counts: vec![],
            edges: vec![],
            edge_weights: None,
            adjacency: None,
        }
    }

    #[test]
    fn tally_keys_accumulates_multiple_keys_and_sparse_district_ids() {
        let graph = graph_with_attr_columns(vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]);

        let (totals, n_districts, observed) = tally_keys(&graph, &[1, 3, 1], &[0, 1]).unwrap();

        assert_eq!(n_districts, 4);
        assert_eq!(observed, (1u128 << 1) | (1u128 << 3));
        assert_eq!(totals, vec![0.0, 4.0, 0.0, 2.0, 0.0, 40.0, 0.0, 20.0]);
    }

    #[test]
    fn incremental_tallies_rejects_delta_old_label_mismatch() {
        let graph = graph_with_attr_columns(vec![vec![1.0, 2.0, 3.0]]);
        let before = vec![1, 1, 2];
        let changes = vec![(1usize, 2u16, 1u16)];
        let mut state = IncrementalTallies::new(&graph, &[0]);

        state.seed(&before).unwrap();
        let err = state.update_delta(&before, &changes).unwrap_err();

        assert!(
            err.to_string().contains("old label mismatch"),
            "unexpected error: {err}",
        );
    }
}
