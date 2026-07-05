use crate::district::{observe_district, validate_district_set_unchanged, MAX_DISTRICTS};
use crate::graph::Graph;
use crate::input::BenSource;
use crate::metrics::twodelta::PostDeltaLabels;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{
    capped_reps, make_progress_bar, parquet_compression, run_pipeline, AssignmentLengthCheck,
    PARQUET_BATCH_ROWS,
};
use ben::io::reader::TwoDeltaFrameEvent;
use ben::BenVariant;
use std::fs::File;
use std::io;

#[inline]
fn reock_score() -> f64 {
    todo!("Implement Reock score calculation based on area and perimeter")
}

fn reock_rows(
    _assignment: &[u16],
    _area_values: &[f64],
    _total_perimeter_values: &[f64],
    _edges: &[(u32, u32)],
    _shared_perimeters: &[f64],
) -> crate::error::Result<(Vec<f64>, u16, u128)> {
    // let mut observed = 0u128;
    // let mut max_district = 0usize;
    // let mut area_by_district = vec![0.0f64; MAX_DISTRICTS as usize];
    // let mut perimeter_by_district = vec![0.0f64; MAX_DISTRICTS as usize];

    todo!("Implement Reock score calculation based on area and perimeter");
}

/// NOTE: Fill this in
struct IncrementalReock<'g> {
    graph: &'g Graph,
    area_values: &'g [f64],
    total_perimeter_values: &'g [f64],
    shared_perimeters: &'g [f64],
    area_by_district: Vec<f64>,
    perimeter_by_district: Vec<f64>,
    node_counts: Vec<u32>,
    observed: u128,
    post_delta_labels: PostDeltaLabels,
    seen_edges: Vec<u64>,
    gen: u64,
}

impl<'g> IncrementalReock<'g> {
    fn new(
        graph: &'g Graph,
        area_values: &'g [f64],
        total_perimeter_values: &'g [f64],
        shared_perimeters: &'g [f64],
    ) -> Self {
        Self {
            graph,
            area_values,
            total_perimeter_values,
            shared_perimeters,
            area_by_district: vec![0.0; MAX_DISTRICTS as usize],
            perimeter_by_district: vec![0.0; MAX_DISTRICTS as usize],
            node_counts: vec![0; MAX_DISTRICTS as usize],
            observed: 0,
            post_delta_labels: PostDeltaLabels::new(graph.node_count),
            seen_edges: vec![0; graph.edges.len()],
            gen: 0,
        }
    }

    /// Recompute all area/perimeter state from a snapshot assignment.
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        self.area_by_district.fill(0.0);
        self.perimeter_by_district.fill(0.0);
        self.node_counts.fill(0);
        self.observed = 0;

        for (node, &district) in assignment.iter().enumerate() {
            observe_district(&mut self.observed, district)?;
            let district = district as usize;
            self.node_counts[district] += 1;
            self.area_by_district[district] += self.area_values[node];
            self.perimeter_by_district[district] += self.total_perimeter_values[node];
        }

        for (edge_index, &(node_u, node_v)) in self.graph.edges.iter().enumerate() {
            let district_u = assignment[node_u as usize] as usize;
            let district_v = assignment[node_v as usize] as usize;
            if district_u == district_v {
                self.perimeter_by_district[district_u] -= 2.0 * self.shared_perimeters[edge_index];
            }
        }

        Ok(())
    }

    /// Apply one delta event to the maintained area/perimeter state and district set.
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
            if old == new {
                continue;
            }

            let old = old as usize;
            let new = new as usize;
            self.node_counts[new] += 1;
            self.node_counts[old] -= 1;
            if self.node_counts[old] == 0 {
                self.observed &= !(1u128 << old);
            }
            self.area_by_district[old] -= self.area_values[node];
            self.area_by_district[new] += self.area_values[node];
            self.perimeter_by_district[old] -= self.total_perimeter_values[node];
            self.perimeter_by_district[new] += self.total_perimeter_values[node];
        }

        self.post_delta_labels.refresh(changes);
        self.gen += 1;
        for &(node, _old, _new) in changes {
            for &(_neighbor, edge_index) in self.graph.neighbors(node) {
                let edge_index = edge_index as usize;
                if self.seen_edges[edge_index] == self.gen {
                    continue;
                }
                self.seen_edges[edge_index] = self.gen;
                let (u, v) = self.graph.edges[edge_index];
                let u = u as usize;
                let v = v as usize;
                let before_u = before[u] as usize;
                let before_v = before[v] as usize;
                let after_u = self.post_delta_labels.label(before, u) as usize;
                let after_v = self.post_delta_labels.label(before, v) as usize;
                let shared_perimeter = self.shared_perimeters[edge_index];
                if before_u == before_v {
                    self.perimeter_by_district[before_u] += 2.0 * shared_perimeter;
                }
                if after_u == after_v {
                    self.perimeter_by_district[after_u] -= 2.0 * shared_perimeter;
                }
            }
        }

        Ok(())
    }

    fn scores(&self) -> crate::error::Result<Vec<f64>> {
        let mut scores = vec![0.0; MAX_DISTRICTS as usize];
        for (district, score) in scores.iter_mut().enumerate() {
            if (self.observed & (1u128 << district)) == 0 {
                continue;
            }
            let perimeter = self.perimeter_by_district[district];
            if perimeter <= 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "district {} has nonpositive perimeter {}; check the area/perimeter/shared-perimeter keys",
                        district, perimeter
                    ),
                )
                .into());
            }
            *score = reock_score();
        }
        Ok(scores)
    }
}

/// Run Reock directly from TwoDelta events, reseeding on snapshots and patching deltas.
#[allow(clippy::too_many_arguments)]
fn run_incremental_twodelta_reock(
    graph: &Graph,
    source: &BenSource,
    writer: &mut DistrictMetricWriter,
    area_values: &[f64],
    total_perimeters: &[f64],
    shared_perimeters: &[f64],
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
    let mut state = IncrementalReock::new(graph, area_values, total_perimeters, shared_perimeters);
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
                    return Err(crate::error::Error::AssignmentLength {
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
            Some(expected) => {
                validate_district_set_unchanged(state.observed, expected, "polsby-popper")?;
            }
        }

        let scores = state.scores()?;
        writer.push_row(step, n_reps as u32, accepted, (state.observed, &scores))?;
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

#[allow(clippy::too_many_arguments)]
pub fn tally_and_save_reock(
    _graph: Graph,
    _source: &BenSource,
    _out_file_name: &str,
    _area_key: &str,
    _perim_key: Option<&str>,
    _boundary_perim_key: Option<&str>,
    _show_progress: bool,
    _max_samples: Option<usize>,
    _high_compression: bool,
) -> crate::error::Result<()> {
    todo!("Implement tally_and_save_reock function to compute and save Reock scores");
}

#[cfg(test)]
mod tests {}
