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
fn polsby_popper_score(area: f64, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        0.0
    } else {
        4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
    }
}

fn derive_total_perimeters(
    boundary_perimeters: &[f64],
    edges: &[(u32, u32)],
    shared_perimeters: &[f64],
) -> Vec<f64> {
    let mut total_perimeters = boundary_perimeters.to_vec();
    for (edge_index, &(node_u, node_v)) in edges.iter().enumerate() {
        total_perimeters[node_u as usize] += shared_perimeters[edge_index];
        total_perimeters[node_v as usize] += shared_perimeters[edge_index];
    }
    total_perimeters
}

fn polsby_popper_rows(
    assignment: &[u16],
    area_values: &[f64],
    total_perimeter_values: &[f64],
    edges: &[(u32, u32)],
    shared_perimeters: &[f64],
) -> crate::error::Result<(Vec<f64>, u16, u128)> {
    let mut observed = 0u128;
    let mut max_district = 0usize;
    let mut area_by_district = vec![0.0f64; MAX_DISTRICTS as usize];
    let mut perimeter_by_district = vec![0.0f64; MAX_DISTRICTS as usize];

    for (node, &district) in assignment.iter().enumerate() {
        observe_district(&mut observed, district)?;
        let district = district as usize;
        area_by_district[district] += area_values[node];
        perimeter_by_district[district] += total_perimeter_values[node];
        max_district = max_district.max(district);
    }
    let n_districts = max_district + 1;

    for (edge_index, &(node_u, node_v)) in edges.iter().enumerate() {
        let district_u = assignment[node_u as usize] as usize;
        let district_v = assignment[node_v as usize] as usize;
        if district_u == district_v {
            perimeter_by_district[district_u] -= 2.0 * shared_perimeters[edge_index];
        }
    }

    // A real district cannot have a nonpositive perimeter; one here means the geometry data is
    // wrong (e.g. a direct --perim-key inconsistent with shared_perim, or perimeter data missing
    // for a district's nodes). Scoring it 0.0 would bury the data problem in plausible-looking
    // output, so fail instead. Unobserved district ids (gaps in the label range) carry 0.0 but are
    // never written, so only observed districts are checked.
    for (district, &perimeter) in perimeter_by_district.iter().enumerate() {
        if (observed & (1u128 << district)) != 0 && perimeter <= 0.0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "district {} has nonpositive perimeter {}; check the area/perimeter/shared-perimeter keys",
                    district, perimeter
                ),
            )
            .into());
        }
    }

    let scores = (0..n_districts)
        .map(|district| {
            polsby_popper_score(area_by_district[district], perimeter_by_district[district])
        })
        .collect();
    Ok((scores, n_districts as u16, observed))
}

/// Maintains Polsby-Popper district area/perimeter state across TwoDelta events.
///
/// `update_delta` expects `before` to still be the pre-delta assignment; the caller applies the
/// changes only after node totals and incident-edge perimeter adjustments have been patched.
struct IncrementalPolsbyPopper<'g> {
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

impl<'g> IncrementalPolsbyPopper<'g> {
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
            *score = polsby_popper_score(self.area_by_district[district], perimeter);
        }
        Ok(scores)
    }
}

/// Run Polsby-Popper directly from TwoDelta events, reseeding on snapshots and patching deltas.
#[allow(clippy::too_many_arguments)]
fn run_incremental_twodelta_polsby_popper(
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
    let mut state =
        IncrementalPolsbyPopper::new(graph, area_values, total_perimeters, shared_perimeters);
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
pub fn tally_and_save_polsby_popper(
    graph: Graph,
    source: &BenSource,
    out_file_name: &str,
    area_key: &str,
    perim_key: Option<&str>,
    boundary_perim_key: Option<&str>,
    show_progress: bool,
    max_samples: Option<usize>,
    high_compression: bool,
) -> crate::error::Result<()> {
    let area_values = graph
        .numeric_column(area_key)
        .unwrap_or_else(|| panic!("area key {:?} not pre-loaded on graph", area_key));

    let shared_perimeters = graph
        .edge_weight_column()
        .unwrap_or_else(|| panic!("shared perimeter edge column not pre-loaded on graph"));

    let total_perimeters = if let Some(perim_key) = perim_key {
        graph
            .numeric_column(perim_key)
            .unwrap_or_else(|| panic!("perimeter key {:?} not pre-loaded on graph", perim_key))
            .to_vec()
    } else {
        let boundary_key = boundary_perim_key
            .expect("boundary perimeter key should exist when direct perimeter key is absent");
        let boundary_perimeters = graph.numeric_column(boundary_key).unwrap_or_else(|| {
            panic!(
                "boundary perimeter key {:?} not pre-loaded on graph",
                boundary_key
            )
        });
        derive_total_perimeters(boundary_perimeters, &graph.edges, shared_perimeters)
    };

    // The writer fixes its district-column schema from the first row's observed set and creates
    // the output file at that point; a run that fails before decoding a plan leaves no file.
    let out_path = out_file_name.to_string();
    let mut writer = DistrictMetricWriter::new(
        Box::new(move || File::create(out_path)),
        parquet_compression(high_compression),
        PARQUET_BATCH_ROWS,
    );

    if source.variant()? == BenVariant::TwoDelta && graph.adjacency.is_some() {
        run_incremental_twodelta_polsby_popper(
            &graph,
            source,
            &mut writer,
            area_values,
            &total_perimeters,
            shared_perimeters,
            show_progress,
            max_samples,
        )?;
    } else {
        run_pipeline(
            source,
            AssignmentLengthCheck::MatchesGraph(graph.node_count),
            // The pipeline enforces a fixed district set, so the schema fixed from the first row
            // holds.
            "polsby-popper",
            |assignment, _n_reps| {
                let (scores, _n_districts, observed) = polsby_popper_rows(
                    assignment,
                    area_values,
                    &total_perimeters,
                    &graph.edges,
                    shared_perimeters,
                )?;
                Ok((observed, (scores, observed)))
            },
            |step, n_reps, accepted, (scores, observed)| {
                writer.push_row(step, n_reps, accepted, (observed, &scores))
            },
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
    use super::{
        derive_total_perimeters, polsby_popper_rows, polsby_popper_score, IncrementalPolsbyPopper,
    };
    use crate::graph::{CsrAdjacency, Graph};
    use std::collections::HashMap;

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
    fn polsby_popper_score_returns_zero_for_nonpositive_perimeter() {
        assert_eq!(polsby_popper_score(10.0, 0.0), 0.0);
        assert_eq!(polsby_popper_score(10.0, -4.0), 0.0);
    }

    #[test]
    fn derive_total_perimeters_adds_boundary_and_shared_lengths() {
        let totals = derive_total_perimeters(
            &[3.0, 2.0, 2.0, 3.0],
            &[(0, 1), (1, 2), (2, 3)],
            &[1.0, 1.0, 1.0],
        );
        assert_eq!(totals, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn polsby_popper_rows_errors_on_nonpositive_district_perimeter() {
        // Zero total perimeters with no edges → both observed districts compute perimeter 0.0,
        // which is physically impossible for a real district and means the geometry keys are
        // wrong. This must error, not score 0.0 into plausible-looking output.
        let err = polsby_popper_rows(&[1, 2], &[1.0, 1.0], &[0.0, 0.0], &[], &[]).unwrap_err();
        assert!(
            err.to_string()
                .contains("district 1 has nonpositive perimeter 0"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn polsby_popper_rows_matches_known_two_district_example() {
        let (scores, n_districts, observed) = polsby_popper_rows(
            &[1, 1, 2, 2],
            &[1.0, 1.0, 1.0, 1.0],
            &[4.0, 4.0, 4.0, 4.0],
            &[(0, 1), (1, 2), (2, 3)],
            &[1.0, 1.0, 1.0],
        )
        .unwrap();

        let expected = 2.0 * std::f64::consts::PI / 9.0;
        assert_eq!(n_districts, 3);
        assert_eq!(observed, (1u128 << 1) | (1u128 << 2));
        assert_eq!(scores[0], 0.0);
        assert!((scores[1] - expected).abs() < 1e-12);
        assert!((scores[2] - expected).abs() < 1e-12);
    }

    #[test]
    fn incremental_polsby_popper_rejects_delta_old_label_mismatch() {
        let graph = graph_with_path_adjacency();
        let before = vec![1, 1, 2, 2];
        let changes = vec![(1usize, 2u16, 1u16)];
        let area = vec![1.0; 4];
        let total_perimeter = vec![4.0; 4];
        let shared_perimeter = vec![1.0; 3];
        let mut state =
            IncrementalPolsbyPopper::new(&graph, &area, &total_perimeter, &shared_perimeter);

        state.seed(&before).unwrap();
        let err = state.update_delta(&before, &changes).unwrap_err();

        assert!(
            err.to_string().contains("old label mismatch"),
            "unexpected error: {err}",
        );
    }
}
