use crate::cli::{build_tally_output_dir, build_tally_output_path};
use crate::district::{observe_district, observed_assignment_districts, MAX_DISTRICTS};
use crate::graph::Graph;
use crate::input::BenSource;
use crate::metrics::twodelta::{
    run_incremental_twodelta, DeltaChange, IncrementalTwoDeltaMetric, TwoDeltaRow,
    TwoDeltaRunOptions,
};
use crate::metrics::{validate_assignment_length, PreparedMetricOutput};
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{
    parquet_compression, run_pipeline, AssignmentLengthCheck, PARQUET_BATCH_ROWS,
};
use ben::BenVariant;
use std::fs::{create_dir_all, File};

/// Hot loop: flat index into pre-parsed attribute columns, accumulate into a flat per-district
/// totals vector. No HashMap work inside the inner loop.
///
/// `totals` is a flat `Vec<f64>` of shape `[n_keys * n_districts]`, where
/// `n_districts = max(assignment) + 1`. `observed` has bit `d` set iff district `d` appeared in
/// this sample's assignment.
fn tally_columns(
    columns: &[Vec<f64>],
    assignment: &[u16],
) -> crate::error::Result<(Vec<f64>, u16, u128)> {
    let (n_districts, observed) = observed_assignment_districts(assignment)?;
    let n_districts = n_districts as usize;
    let mut totals = vec![0.0f64; columns.len() * n_districts];
    for (key_index, column) in columns.iter().enumerate() {
        let offset = key_index * n_districts;
        for (node_index, &value) in column.iter().enumerate() {
            totals[offset + assignment[node_index] as usize] += value;
        }
    }
    Ok((totals, n_districts as u16, observed))
}

#[derive(Debug)]
pub struct PreparedTally {
    columns: Vec<Vec<f64>>,
    node_count: usize,
}

impl PreparedTally {
    pub fn new(columns: Vec<Vec<f64>>) -> crate::error::Result<Self> {
        let node_count = columns.first().map_or(0, Vec::len);
        if columns.is_empty() {
            return Err(crate::error::invalid_data(
                "a prepared tally requires at least one numeric column",
            )
            .into());
        }
        for (index, column) in columns.iter().enumerate() {
            if column.len() != node_count {
                return Err(crate::error::Error::AssignmentLength {
                    actual: column.len(),
                    actual_label: "tally column length",
                    expected: node_count,
                    expected_label: "graph node count",
                });
            }
            if column.iter().any(|value| !value.is_finite()) {
                return Err(crate::error::invalid_data(format!(
                    "tally column {index} contains a non-finite value"
                ))
                .into());
            }
        }
        Ok(Self {
            columns,
            node_count,
        })
    }

    pub fn score_assignment(
        &self,
        assignment: &[u16],
    ) -> crate::error::Result<PreparedMetricOutput> {
        validate_assignment_length(
            assignment,
            self.node_count,
            "graph node count",
            "assignment length",
        )?;
        self.score_checked_assignment(assignment)
    }

    fn score_checked_assignment(
        &self,
        assignment: &[u16],
    ) -> crate::error::Result<PreparedMetricOutput> {
        let (values, district_slots, observed) = tally_columns(&self.columns, assignment)?;
        Ok(PreparedMetricOutput {
            values,
            table_count: self.columns.len(),
            district_slots: district_slots as usize,
            observed,
        })
    }
}

/// Maintains per-key district totals across TwoDelta events.
///
/// `update_delta` expects `before` to still be the pre-delta assignment; the caller applies the
/// changes after the totals and district counts are patched.
struct IncrementalTallies<'g> {
    metric: &'g PreparedTally,
    totals: Vec<f64>,
    node_counts: Vec<u32>,
    observed: u128,
}

impl<'g> IncrementalTallies<'g> {
    fn new(metric: &'g PreparedTally) -> Self {
        Self {
            metric,
            totals: vec![0.0; metric.columns.len() * MAX_DISTRICTS as usize],
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
            for (key_index, column) in self.metric.columns.iter().enumerate() {
                let offset = key_index * MAX_DISTRICTS as usize;
                self.totals[offset + district as usize] += column[node];
            }
        }

        Ok(())
    }

    /// Apply one delta event to the maintained tallies and district set.
    fn update_delta(
        &mut self,
        _before: &[u16],
        changes: &[DeltaChange],
    ) -> crate::error::Result<()> {
        for change in changes {
            observe_district(&mut self.observed, change.old)?;
            observe_district(&mut self.observed, change.new)?;
            self.node_counts[change.new as usize] += 1;
            self.node_counts[change.old as usize] -= 1;
            if self.node_counts[change.old as usize] == 0 {
                self.observed &= !(1u128 << change.old);
            }

            for (key_index, column) in self.metric.columns.iter().enumerate() {
                let value = column[change.node];
                let offset = key_index * MAX_DISTRICTS as usize;
                self.totals[offset + change.old as usize] -= value;
                self.totals[offset + change.new as usize] += value;
            }
        }

        Ok(())
    }
}

impl IncrementalTwoDeltaMetric for IncrementalTallies<'_> {
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        IncrementalTallies::seed(self, assignment)
    }

    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[DeltaChange],
    ) -> crate::error::Result<()> {
        IncrementalTallies::update_delta(self, before, changes)
    }

    fn observed(&self) -> u128 {
        self.observed
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
    let metric = PreparedTally::new(
        attr_column_indices
            .iter()
            .map(|&index| graph.attr_columns[index].clone())
            .collect(),
    )?;

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
        let mut state = IncrementalTallies::new(&metric);
        run_incremental_twodelta(
            source,
            TwoDeltaRunOptions {
                expected_len: graph.node_count,
                expected_len_label: "graph node count",
                output_name: "tally",
                show_progress,
                max_samples,
            },
            &mut state,
            |state,
             TwoDeltaRow {
                 step,
                 n_reps,
                 accepted,
             }| {
                push_tally_rows(
                    &mut writers,
                    step,
                    n_reps,
                    accepted,
                    state.observed,
                    &state.totals,
                    MAX_DISTRICTS as usize,
                )
            },
        )?;
    } else {
        run_pipeline(
            source,
            AssignmentLengthCheck::Exact {
                expected: graph.node_count,
                label: "graph node count",
            },
            // The pipeline enforces that the district set is identical for every plan, so the
            // schema each writer fixes from its first row holds for the whole run.
            "tally",
            |assignment, _n_reps| {
                let output = metric.score_checked_assignment(assignment)?;
                Ok((
                    output.observed,
                    (output.values, output.district_slots, output.observed),
                ))
            },
            |step, n_reps, accepted, (totals, n_districts, observed)| {
                push_tally_rows(
                    &mut writers,
                    step,
                    n_reps,
                    accepted,
                    observed,
                    &totals,
                    n_districts,
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
    use super::{tally_columns, DeltaChange, IncrementalTallies, PreparedTally};
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

        let (totals, n_districts, observed) =
            tally_columns(&graph.attr_columns, &[1, 3, 1]).unwrap();

        assert_eq!(n_districts, 4);
        assert_eq!(observed, (1u128 << 1) | (1u128 << 3));
        assert_eq!(totals, vec![0.0, 4.0, 0.0, 2.0, 0.0, 40.0, 0.0, 20.0]);

        let metric = PreparedTally::new(graph.attr_columns.clone()).unwrap();
        let output = metric.score_assignment(&[1, 3, 1]).unwrap();
        assert_eq!(output.table(0), Some(&[0.0, 4.0, 0.0, 2.0][..]));
        assert_eq!(output.table(1), Some(&[0.0, 40.0, 0.0, 20.0][..]));
    }

    #[test]
    fn incremental_tallies_updates_delta() {
        let graph = graph_with_attr_columns(vec![vec![1.0, 2.0, 3.0]]);
        let before = vec![1, 1, 2];
        let changes = vec![DeltaChange {
            node: 1,
            old: 1,
            new: 2,
        }];
        let metric = PreparedTally::new(graph.attr_columns.clone()).unwrap();
        let mut state = IncrementalTallies::new(&metric);

        state.seed(&before).unwrap();
        state.update_delta(&before, &changes).unwrap();

        assert_eq!(state.totals[1], 1.0);
        assert_eq!(state.totals[2], 5.0);
        assert_eq!(state.observed, (1u128 << 1) | (1u128 << 2));
    }
}
