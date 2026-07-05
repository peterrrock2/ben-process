use crate::district::{observe_district, MAX_DISTRICTS};
use crate::geometry::{Circle, ReockGeometries};
use crate::graph::Graph;
use crate::input::BenSource;
use crate::metrics::twodelta::PostDeltaLabels;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{
    parquet_compression, run_pipeline, AssignmentLengthCheck, PARQUET_BATCH_ROWS,
};
// use ben::io::reader::TwoDeltaFrameEvent;
use ben::BenVariant;
use geo::{ConvexHull, Coord, Point};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io;

const EPS: f64 = 1e-9;

#[inline]
fn l2_sq_dist(p: Coord, q: Coord) -> f64 {
    (p.x - q.x).powi(2) + (p.y - q.y).powi(2)
}

fn diameter_circle(p: Coord, q: Coord) -> Circle {
    let center = Coord {
        x: (p.x + q.x) / 2.0,
        y: (p.y + q.y) / 2.0,
    };
    let radius = l2_sq_dist(p, q).sqrt() / 2.0;
    Circle { center, radius }
}

fn largest_diameter_circle(p: Coord, q: Coord, r: Coord) -> Circle {
    let points = vec![p, q, r];
    let (point1, point2) = {
        let mut ret = (points[0], points[1]);
        for (p1, p2) in [
            (points[0], points[1]),
            (points[0], points[2]),
            (points[1], points[2]),
        ] {
            if l2_sq_dist(p1, p2) > l2_sq_dist(ret.0, ret.1) {
                ret = (p1, p2);
            }
        }
        ret
    };

    diameter_circle(point1, point2)
}

#[inline]
fn determinant(p: Coord, q: Coord, r: Coord) -> f64 {
    p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)
}

fn circumcircle(p: Coord, q: Coord, r: Coord) -> Option<Circle> {
    let det = determinant(p, q, r);

    // Check collinearity
    if det.abs() < EPS {
        return None;
    }

    let d = 2.0 * det;
    let ux = ((p.x.powi(2) + p.y.powi(2)) * (q.y - r.y)
        + (q.x.powi(2) + q.y.powi(2)) * (r.y - p.y)
        + (r.x.powi(2) + r.y.powi(2)) * (p.y - q.y))
        / d;
    let uy = ((p.x.powi(2) + p.y.powi(2)) * (r.x - q.x)
        + (q.x.powi(2) + q.y.powi(2)) * (p.x - r.x)
        + (r.x.powi(2) + r.y.powi(2)) * (q.x - p.x))
        / d;
    let center = Coord { x: ux, y: uy };
    let radius = ((center.x - p.x).powi(2) + (center.y - p.y).powi(2)).sqrt();
    Some(Circle { center, radius })
}

fn in_circle(p: Coord, circle: &Circle) -> bool {
    let dist_sq = l2_sq_dist(p, circle.center);
    dist_sq <= circle.radius.powi(2) + EPS
}

// Use Welzl's algorithm to compute the minimum enclosing circle of a set of points
fn compute_minimum_enclosing_circle_area(hull: &geo::Polygon<f64>) -> Option<f64> {
    let mut points: Vec<Coord<f64>> = hull.exterior().points().map(|p: Point| p.0).collect();

    if points.len() < 2 {
        return None;
    }

    // Remove the last point if it is the same as the first point (closed polygon)
    if points[0] == points[points.len() - 1] {
        points.pop();
    }

    if points.len() < 2 {
        return None;
    }

    let mut rng = rand::rng();
    points.shuffle(&mut rng);

    let p0 = points[0];
    let p1 = points[1];

    let mut mec = diameter_circle(p0, p1);
    for i in 0..points.len() {
        if in_circle(points[i], &mec) {
            continue;
        }

        mec = Circle {
            center: points[i],
            radius: 0.0,
        };

        for j in 0..i {
            if in_circle(points[j], &mec) {
                continue;
            }

            mec = diameter_circle(points[i], points[j]);

            for k in 0..j {
                if in_circle(points[k], &mec) {
                    continue;
                }

                mec = circumcircle(points[i], points[j], points[k])
                    .unwrap_or_else(|| largest_diameter_circle(points[i], points[j], points[k]));
            }
        }
    }

    Some(std::f64::consts::PI * mec.radius.powi(2))
}

/// Compute the Reock score for a district given its point cloud and convex hull area.
fn reock_score(point_cloud: Vec<Coord>, hull_area: f64) -> crate::error::Result<f64> {
    let hull = geo::MultiPoint::from(point_cloud).convex_hull();
    let mec_area = compute_minimum_enclosing_circle_area(&hull).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Failed to compute minimum enclosing circle area for one of the districts.",
        )
    })?;

    Ok(if mec_area > 0.0 {
        hull_area / mec_area
    } else {
        0.0
    })
}

fn reock_rows(
    assignment: &[u16],
    reock_geometries: &ReockGeometries,
) -> crate::error::Result<(Vec<f64>, u16, u128)> {
    let mut observed = 0u128;
    let mut area_by_district = vec![0.0f64; MAX_DISTRICTS as usize];
    let mut district_hull_points = HashMap::<u16, Vec<Coord<f64>>>::new();
    let mut max_district = 0u16;

    if assignment.len() != reock_geometries.units.len() {
        return Err(crate::error::Error::AssignmentLength {
            actual: assignment.len(),
            expected: reock_geometries.units.len(),
        });
    }

    for (idx, &district) in assignment.iter().enumerate() {
        observe_district(&mut observed, district)?;

        let current_unit = &reock_geometries.units[idx];
        area_by_district[district as usize] += current_unit.area;

        district_hull_points
            .entry(district)
            .or_default()
            .extend(current_unit.convex_hull_points.iter().copied());

        max_district = max_district.max(district);
    }

    let mut scores = vec![0.0; MAX_DISTRICTS as usize];

    for (district, points) in district_hull_points {
        scores[district as usize] = reock_score(points, area_by_district[district as usize])?;
    }

    Ok((scores, max_district + 1, observed))
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
        todo!();
        // let mut scores = vec![0.0; MAX_DISTRICTS as usize];
        // for (district, score) in scores.iter_mut().enumerate() {
        //     if (self.observed & (1u128 << district)) == 0 {
        //         continue;
        //     }
        //     let perimeter = self.perimeter_by_district[district];
        //     if perimeter <= 0.0 {
        //         return Err(io::Error::new(
        //             io::ErrorKind::InvalidData,
        //             format!(
        //                 "district {} has nonpositive perimeter {}; check the
        // area/perimeter/shared-perimeter keys",                 district, perimeter
        //             ),
        //         )
        //         .into());
        //     }
        //     *score = reock_score();
        // }
        // Ok(scores)
    }
}

/// Run Reock directly from TwoDelta events, reseeding on snapshots and patching deltas.
fn run_incremental_twodelta_reock(
    _reock_geometries: ReockGeometries,
    _source: &BenSource,
    _writer: &mut DistrictMetricWriter,
    _show_progress: bool,
    _max_samples: Option<usize>,
) -> crate::error::Result<()> {
    todo!();
    // let progress_bar = if show_progress {
    //     Some(make_progress_bar(match max_samples {
    //         Some(n) => n,
    //         None => source.count_samples()?,
    //     }))
    // } else {
    //     None
    // };
    //
    // let mut remaining_samples = max_samples;
    // let mut assignment: Option<Vec<u16>> = None;
    // let mut expected_observed: Option<u128> = None;
    // let mut state = IncrementalReock::new(graph, area_values, total_perimeters,
    // shared_perimeters); let mut step = 1u64;
    //
    // for (accepted, event) in (1u64..).zip(source.open_reader()?.into_twodelta_events()) {
    //     if remaining_samples == Some(0) {
    //         break;
    //     }
    //
    //     let n_reps = match event? {
    //         TwoDeltaFrameEvent::Snapshot {
    //             assignment: snapshot,
    //             count,
    //             ..
    //         } => {
    //             if snapshot.len() != graph.node_count {
    //                 return Err(crate::error::Error::AssignmentLength {
    //                     actual: snapshot.len(),
    //                     expected: graph.node_count,
    //                 });
    //             }
    //             state.seed(&snapshot)?;
    //             assignment = Some(snapshot);
    //             capped_reps(&mut remaining_samples, count)
    //         }
    //         TwoDeltaFrameEvent::Delta { changes, count } => {
    //             let assignment = assignment.as_mut().ok_or_else(|| {
    //                 io::Error::new(
    //                     io::ErrorKind::InvalidData,
    //                     "TwoDelta delta event appeared before an initial snapshot",
    //                 )
    //             })?;
    //             let changes = changes
    //                 .into_iter()
    //                 .map(|(node, old, new)| (node as usize, old, new))
    //                 .collect::<Vec<_>>();
    //             state.update_delta(assignment, &changes)?;
    //             for (node, _old, new) in changes {
    //                 assignment[node] = new;
    //             }
    //             capped_reps(&mut remaining_samples, count)
    //         }
    //     };
    //
    //     match expected_observed {
    //         None => expected_observed = Some(state.observed),
    //         Some(expected) => {
    //             validate_district_set_unchanged(state.observed, expected, "polsby-popper")?;
    //         }
    //     }
    //
    //     let scores = state.scores()?;
    //     writer.push_row(step, n_reps as u32, accepted, (state.observed, &scores))?;
    //     step += n_reps as u64;
    //     if let Some(progress_bar) = &progress_bar {
    //         progress_bar.inc(n_reps as u64);
    //     }
    // }
    //
    // if let Some(progress_bar) = progress_bar {
    //     progress_bar.finish_and_clear();
    // }
    // Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn tally_and_save_reock(
    reock_geometries: ReockGeometries,
    source: &BenSource,
    out_file_name: &str,
    show_progress: bool,
    max_samples: Option<usize>,
    high_compression: bool,
) -> crate::error::Result<()> {
    // The writer fixes its district-column schema from the first row's observed set and creates
    // the output file at that point; a run that fails before decoding a plan leaves no file.
    let out_path = out_file_name.to_string();
    let mut writer = DistrictMetricWriter::new(
        Box::new(move || File::create(out_path)),
        parquet_compression(high_compression),
        PARQUET_BATCH_ROWS,
    );

    if source.variant()? == BenVariant::TwoDelta {
        run_incremental_twodelta_reock(
            reock_geometries,
            source,
            &mut writer,
            show_progress,
            max_samples,
        )?;
    } else {
        run_pipeline(
            source,
            AssignmentLengthCheck::MatchesGeometryFile(reock_geometries.units.len()),
            // The pipeline enforces a fixed district set, so the schema fixed from the first row
            // holds.
            "reock",
            // process
            |assignment, _n_reps| {
                let (scores, _n_districts, observed) = reock_rows(assignment, &reock_geometries)?;
                Ok((observed, (scores, observed)))
            },
            // on row
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
mod tests {}
