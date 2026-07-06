use crate::district::{observe_district, validate_district_set_unchanged, MAX_DISTRICTS};
use crate::geometry::ReockGeometries;
use crate::input::BenSource;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{
    capped_reps, make_progress_bar, parquet_compression, run_pipeline, AssignmentLengthCheck,
    PARQUET_BATCH_ROWS,
};
use ben::io::reader::TwoDeltaFrameEvent;
use ben::BenVariant;
use geo::Coord;
use std::fs::File;
use std::io;

const EPS: f64 = 1e-9;
const REOCK_SHUFFLE_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

#[derive(Debug, Clone, Copy)]
struct Circle {
    center: Coord<f64>,
    radius: f64,
}

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
    let mut endpoints = (p, q);
    for pair in [(p, r), (q, r)] {
        if l2_sq_dist(pair.0, pair.1) > l2_sq_dist(endpoints.0, endpoints.1) {
            endpoints = pair;
        }
    }

    diameter_circle(endpoints.0, endpoints.1)
}

#[inline]
fn determinant(p: Coord, q: Coord, r: Coord) -> f64 {
    p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)
}

fn circumcircle(p: Coord, q: Coord, r: Coord) -> Option<Circle> {
    let det = determinant(p, q, r);

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

// Randomized incremental minimum enclosing circle over a point cloud.
fn compute_minimum_enclosing_circle_area(
    points: &mut [Coord<f64>],
    rng: &mut fastrand::Rng,
) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    rng.shuffle(points);

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
fn reock_score(
    point_cloud: &mut [Coord<f64>],
    hull_area: f64,
    rng: &mut fastrand::Rng,
) -> crate::error::Result<f64> {
    let mec_area = compute_minimum_enclosing_circle_area(point_cloud, rng).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "Failed to compute minimum enclosing circle area for one of the districts.",
        )
    })?;

    if !hull_area.is_finite() || hull_area <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("district has invalid area {hull_area}"),
        )
        .into());
    }

    if !mec_area.is_finite() || mec_area <= 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("district has invalid minimum enclosing circle area {mec_area}"),
        )
        .into());
    }

    let score = hull_area / mec_area;
    if score > 1.0 + EPS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("district has impossible Reock score {score}"),
        )
        .into());
    }

    Ok(score.min(1.0))
}

struct ReockState {
    scores: Vec<f64>,
    observed: u128,
    area_by_district: Vec<f64>,
    node_counts: Vec<u32>,
    nodes_by_district: Vec<Vec<usize>>,
    node_position: Vec<usize>,
}

fn remove_node(
    node_position: &mut Vec<usize>,
    nodes_by_district: &mut Vec<Vec<usize>>,
    district: usize,
    node: usize,
) -> crate::error::Result<()> {
    let pos = node_position[node];
    let nodes = &mut nodes_by_district[district];
    let Some(&removed) = nodes.get(pos) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("node position for node {node} points outside district {district}'s node list"),
        )
        .into());
    };

    if removed != node {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "node position for node {node} points at node {removed} in district {district}"
            ),
        )
        .into());
    }

    nodes.swap_remove(pos);
    if pos < nodes.len() {
        let moved = nodes[pos];
        node_position[moved] = pos;
    }

    Ok(())
}

fn add_node(
    node_position: &mut Vec<usize>,
    nodes_by_district: &mut Vec<Vec<usize>>,
    district: usize,
    node: usize,
) {
    node_position[node] = nodes_by_district[district].len();
    nodes_by_district[district].push(node);
}

fn reock_rows(
    assignment: &[u16],
    reock_geometries: &ReockGeometries,
) -> crate::error::Result<ReockState> {
    let mut observed = 0u128;
    let mut area_by_district = vec![0.0f64; MAX_DISTRICTS as usize];
    let mut district_hull_points = vec![Vec::new(); MAX_DISTRICTS as usize];
    let mut node_counts = vec![0u32; MAX_DISTRICTS as usize];
    let mut nodes_by_district = vec![Vec::new(); MAX_DISTRICTS as usize];
    let mut node_position = vec![0usize; reock_geometries.units.len()];
    let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);

    if assignment.len() != reock_geometries.units.len() {
        return Err(crate::error::Error::AssignmentLength {
            actual: assignment.len(),
            expected: reock_geometries.units.len(),
        });
    }

    for (idx, &district) in assignment.iter().enumerate() {
        observe_district(&mut observed, district)?;
        node_counts[district as usize] += 1;
        add_node(
            &mut node_position,
            &mut nodes_by_district,
            district as usize,
            idx,
        );

        let current_unit = &reock_geometries.units[idx];
        area_by_district[district as usize] += current_unit.area;

        district_hull_points[district as usize]
            .extend(current_unit.convex_hull_points.iter().copied());
    }

    let mut scores = vec![0.0; MAX_DISTRICTS as usize];

    for (district, mut points) in district_hull_points.into_iter().enumerate() {
        if node_counts[district] == 0 {
            continue;
        }

        scores[district] = reock_score(&mut points, area_by_district[district], &mut rng)?;
    }

    Ok(ReockState {
        scores,
        observed,
        area_by_district,
        node_counts,
        nodes_by_district,
        node_position,
    })
}

struct IncrementalReock<'g> {
    reock_geometries: &'g ReockGeometries,
    state: ReockState,
    hull_point_scratch: Vec<Coord<f64>>,
    rng: fastrand::Rng,
}

impl<'g> IncrementalReock<'g> {
    fn new(reock_geometries: &'g ReockGeometries) -> Self {
        Self {
            reock_geometries,
            state: ReockState {
                scores: vec![0.0; MAX_DISTRICTS as usize],
                observed: 0,
                area_by_district: vec![0.0; MAX_DISTRICTS as usize],
                node_counts: vec![0u32; MAX_DISTRICTS as usize],
                nodes_by_district: vec![Vec::new(); MAX_DISTRICTS as usize],
                node_position: vec![0usize; reock_geometries.units.len()],
            },
            hull_point_scratch: Vec::new(),
            rng: fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED),
        }
    }

    fn remove_node_from_district(
        &mut self,
        node: usize,
        district: u16,
    ) -> crate::error::Result<()> {
        remove_node(
            &mut self.state.node_position,
            &mut self.state.nodes_by_district,
            district as usize,
            node,
        )?;
        self.state.node_counts[district as usize] -= 1;
        self.state.area_by_district[district as usize] -= self.reock_geometries.units[node].area;
        Ok(())
    }

    fn add_node_to_district(&mut self, node: usize, district: u16) {
        add_node(
            &mut self.state.node_position,
            &mut self.state.nodes_by_district,
            district as usize,
            node,
        );
        self.state.node_counts[district as usize] += 1;
        self.state.area_by_district[district as usize] += self.reock_geometries.units[node].area;
    }

    /// Recompute all state from a snapshot assignment.
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        self.state = reock_rows(assignment, self.reock_geometries)?;
        Ok(())
    }

    fn recompute_score(&mut self, district: usize) -> crate::error::Result<()> {
        let points = self.state.nodes_by_district.get(district).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("District {district} not found in nodes_by_district"),
            )
        })?;

        self.hull_point_scratch.clear();
        for &node in points {
            let current_unit = &self.reock_geometries.units[node];
            self.hull_point_scratch
                .extend_from_slice(&current_unit.convex_hull_points);
        }

        self.state.scores[district] = reock_score(
            &mut self.hull_point_scratch,
            self.state.area_by_district[district],
            &mut self.rng,
        )?;

        Ok(())
    }

    /// Apply one delta event to the maintained area, membership, and score state.
    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[(usize, u16, u16)],
    ) -> crate::error::Result<()> {
        let mut touched = Vec::new();

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

            observe_district(&mut self.state.observed, old)?;

            if old == new {
                continue;
            }
            observe_district(&mut self.state.observed, new)?;

            let old = old as usize;
            let new = new as usize;

            self.remove_node_from_district(node, old as u16)?;
            self.add_node_to_district(node, new as u16);

            touched.push(old);
            touched.push(new);
        }

        touched.sort_unstable();
        touched.dedup();

        for district in touched {
            if self.state.node_counts[district] == 0 {
                self.state.scores[district] = 0.0;
                continue;
            }

            self.recompute_score(district)?;
        }

        Ok(())
    }

    fn scores(&self) -> Vec<f64> {
        self.state.scores.clone()
    }

    fn observed(&self) -> u128 {
        self.state.observed
    }
}

/// Run Reock directly from TwoDelta events, reseeding on snapshots and patching deltas.
fn run_incremental_twodelta_reock(
    reock_geometries: &ReockGeometries,
    source: &BenSource,
    writer: &mut DistrictMetricWriter,
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
    let mut inc_state = IncrementalReock::new(&reock_geometries);
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
                if snapshot.len() != reock_geometries.units.len() {
                    return Err(crate::error::Error::AssignmentLength {
                        actual: snapshot.len(),
                        expected: reock_geometries.units.len(),
                    });
                }
                inc_state.seed(&snapshot)?;
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
                inc_state.update_delta(assignment, &changes)?;
                for (node, _old, new) in changes {
                    assignment[node] = new;
                }
                capped_reps(&mut remaining_samples, count)
            }
        };

        match expected_observed {
            None => expected_observed = Some(inc_state.observed()),
            Some(expected) => {
                validate_district_set_unchanged(inc_state.observed(), expected, "reock")?;
            }
        }

        let scores = inc_state.scores();
        writer.push_row(
            step,
            n_reps as u32,
            accepted,
            (inc_state.observed(), &scores),
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
            &reock_geometries,
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
                let state = reock_rows(assignment, &reock_geometries)?;

                Ok((state.observed, (state.scores, state.observed)))
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
mod tests {
    use super::*;
    use crate::geometry::ReockUnit;

    fn coord(x: f64, y: f64) -> Coord<f64> {
        Coord { x, y }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert_close_eps(actual, expected, 1e-9);
    }

    fn assert_close_eps(actual: f64, expected: f64, eps: f64) {
        assert!(
            (actual - expected).abs() <= eps,
            "expected {expected}, got {actual}",
        );
    }

    fn mec_area_for_points(mut points: Vec<Coord<f64>>) -> Option<f64> {
        let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);
        compute_minimum_enclosing_circle_area(&mut points, &mut rng)
    }

    /// Exhaustively checks every 2-point diameter circle and every 3-point circumcircle.
    ///
    /// A minimum enclosing circle is defined by either two boundary points or three boundary
    /// points, so this slow oracle is useful for small test inputs.
    fn brute_force_mec_area(points: &[Coord<f64>]) -> Option<f64> {
        let mut unique = Vec::new();
        for &point in points {
            if !unique.iter().any(|&seen| l2_sq_dist(seen, point) <= EPS) {
                unique.push(point);
            }
        }
        if unique.len() < 2 {
            return None;
        }

        let mut best: Option<Circle> = None;
        for i in 0..unique.len() {
            for j in (i + 1)..unique.len() {
                let circle = diameter_circle(unique[i], unique[j]);
                if unique.iter().all(|&point| in_circle(point, &circle)) {
                    best = Some(match best {
                        Some(best) if best.radius <= circle.radius => best,
                        _ => circle,
                    });
                }
            }
        }

        for i in 0..unique.len() {
            for j in (i + 1)..unique.len() {
                for k in (j + 1)..unique.len() {
                    let Some(circle) = circumcircle(unique[i], unique[j], unique[k]) else {
                        continue;
                    };
                    if unique.iter().all(|&point| in_circle(point, &circle)) {
                        best = Some(match best {
                            Some(best) if best.radius <= circle.radius => best,
                            _ => circle,
                        });
                    }
                }
            }
        }

        best.map(|circle| std::f64::consts::PI * circle.radius.powi(2))
    }

    fn square_points(x: f64, y: f64) -> Vec<Coord<f64>> {
        vec![
            coord(x, y),
            coord(x + 1.0, y),
            coord(x + 1.0, y + 1.0),
            coord(x, y + 1.0),
        ]
    }

    fn square_unit(x: f64, y: f64) -> ReockUnit {
        ReockUnit {
            area: 1.0,
            convex_hull_points: square_points(x, y),
        }
    }

    fn row_test_geometry() -> ReockGeometries {
        ReockGeometries {
            units: vec![
                square_unit(0.0, 0.0),
                square_unit(1.0, 0.0),
                square_unit(10.0, 0.0),
                square_unit(11.0, 0.0),
            ],
        }
    }

    fn incremental_test_geometry() -> ReockGeometries {
        ReockGeometries {
            units: (0..6)
                .map(|node| square_unit((node * 2) as f64, (node % 2) as f64))
                .collect(),
        }
    }

    fn assert_node_positions_valid(state: &ReockState) {
        for (district, nodes) in state.nodes_by_district.iter().enumerate() {
            for (position, &node) in nodes.iter().enumerate() {
                assert_eq!(
                    state.node_position[node], position,
                    "bad position for node {node} in district {district}",
                );
            }
        }
    }

    fn assert_states_match(actual: &ReockState, expected: &ReockState) {
        assert_eq!(actual.observed, expected.observed);
        assert_eq!(actual.node_counts, expected.node_counts);

        for district in 0..MAX_DISTRICTS as usize {
            assert_close(
                actual.area_by_district[district],
                expected.area_by_district[district],
            );
            assert_close_eps(actual.scores[district], expected.scores[district], 1e-8);

            let mut actual_nodes = actual.nodes_by_district[district].clone();
            let mut expected_nodes = expected.nodes_by_district[district].clone();
            actual_nodes.sort_unstable();
            expected_nodes.sort_unstable();
            assert_eq!(actual_nodes, expected_nodes);
        }

        assert_node_positions_valid(actual);
    }

    #[test]
    fn mec_unit_square_uses_diagonal_circle() {
        let area = mec_area_for_points(square_points(0.0, 0.0)).unwrap();

        assert_close(area, std::f64::consts::PI / 2.0);
    }

    #[test]
    fn mec_right_triangle_uses_hypotenuse_circle() {
        let area =
            mec_area_for_points(vec![coord(0.0, 0.0), coord(4.0, 0.0), coord(0.0, 3.0)]).unwrap();

        assert_close(area, std::f64::consts::PI * 6.25);
    }

    #[test]
    fn mec_collinear_points_use_largest_diameter() {
        let area =
            mec_area_for_points(vec![coord(0.0, 0.0), coord(0.0, 2.0), coord(0.0, 5.0)]).unwrap();

        assert_close(area, std::f64::consts::PI * 6.25);
    }

    #[test]
    fn mec_returns_none_for_less_than_two_distinct_points() {
        assert!(mec_area_for_points(vec![coord(1.0, 1.0)]).is_none());
    }

    #[test]
    fn mec_matches_brute_force_for_known_point_clouds() {
        let clouds = [
            vec![coord(0.0, 0.0), coord(2.0, 0.0)],
            vec![
                coord(0.0, 0.0),
                coord(2.0, 0.0),
                coord(1.0, 1.7320508075688772),
            ],
            vec![coord(0.0, 0.0), coord(4.0, 0.0), coord(1.0, 1.0)],
            square_points(0.0, 0.0),
            vec![
                coord(-2.0, -1.0),
                coord(3.0, -1.0),
                coord(4.0, 2.0),
                coord(0.0, 5.0),
                coord(-3.0, 2.0),
            ],
            vec![
                coord(0.0, 0.0),
                coord(2.0, 0.0),
                coord(2.0, 0.0),
                coord(5.0, 0.0),
            ],
        ];

        for points in clouds {
            let expected = brute_force_mec_area(&points).unwrap();
            let actual = mec_area_for_points(points).unwrap();
            assert_close_eps(actual, expected, 1e-8);
        }
    }

    #[test]
    fn mec_matches_brute_force_for_generated_point_clouds() {
        let mut seed = 0x5eed_cafe_u64;

        for case in 0..150 {
            let len = 2 + (case % 12);
            let mut points = Vec::new();
            for _ in 0..len {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let x = ((seed >> 32) % 2000) as f64 / 10.0 - 100.0;
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let y = ((seed >> 32) % 2000) as f64 / 10.0 - 100.0;
                points.push(coord(x, y));
            }

            let expected = brute_force_mec_area(&points).unwrap();
            let actual = mec_area_for_points(points).unwrap();
            assert_close_eps(actual, expected, 1e-7);
        }
    }

    #[test]
    fn reock_score_unit_square_is_two_over_pi() {
        let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);
        let score = reock_score(&mut square_points(0.0, 0.0), 1.0, &mut rng).unwrap();

        assert_close(score, 2.0 / std::f64::consts::PI);
    }

    #[test]
    fn reock_score_rejects_invalid_area() {
        let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);
        assert!(reock_score(&mut square_points(0.0, 0.0), 0.0, &mut rng).is_err());
        assert!(reock_score(&mut square_points(0.0, 0.0), f64::NAN, &mut rng).is_err());
    }

    #[test]
    fn reock_score_rejects_impossible_score_above_one() {
        let mut rng = fastrand::Rng::with_seed(REOCK_SHUFFLE_SEED);
        let err = reock_score(&mut square_points(0.0, 0.0), 2.0, &mut rng).unwrap_err();

        assert!(err.to_string().contains("impossible Reock score"));
    }

    #[test]
    fn reock_rows_scores_hand_computable_assignment() {
        let geometry = row_test_geometry();

        let state = reock_rows(&[0, 0, 1, 1], &geometry).unwrap();

        assert_eq!(state.observed, 0b11);
        assert_eq!(state.node_counts[0], 2);
        assert_eq!(state.node_counts[1], 2);
        assert_close(state.area_by_district[0], 2.0);
        assert_close(state.area_by_district[1], 2.0);
        assert_close(state.scores[0], 8.0 / (5.0 * std::f64::consts::PI));
        assert_close(state.scores[1], 8.0 / (5.0 * std::f64::consts::PI));
        assert_close(state.scores[2], 0.0);
        assert_node_positions_valid(&state);
    }

    #[test]
    fn reock_rows_rejects_assignment_length_mismatch() {
        let Err(err) = reock_rows(&[0, 1, 2], &row_test_geometry()) else {
            panic!("expected assignment length mismatch");
        };

        assert!(matches!(err, crate::error::Error::AssignmentLength { .. }));
    }

    #[test]
    fn reock_rows_rejects_districts_outside_bitmask_limit() {
        let Err(err) = reock_rows(&[MAX_DISTRICTS, 0, 1, 1], &row_test_geometry()) else {
            panic!("expected district limit error");
        };

        assert!(matches!(
            err,
            crate::error::Error::DistrictLimitExceeded { .. }
        ));
    }

    #[test]
    fn incremental_delta_matches_full_recompute_after_moves() {
        let geometry = incremental_test_geometry();
        let mut assignment = vec![0, 0, 0, 1, 1, 1];
        let mut incremental = IncrementalReock::new(&geometry);
        incremental.seed(&assignment).unwrap();

        let changes = vec![(1, 0, 1), (4, 1, 0), (2, 0, 0)];
        incremental.update_delta(&assignment, &changes).unwrap();
        for &(node, _old, new) in &changes {
            assignment[node] = new;
        }

        let expected = reock_rows(&assignment, &geometry).unwrap();
        assert_states_match(&incremental.state, &expected);
    }

    #[test]
    fn incremental_delta_rejects_bad_change_records() {
        let geometry = incremental_test_geometry();
        let assignment = vec![0, 0, 0, 1, 1, 1];
        let mut incremental = IncrementalReock::new(&geometry);
        incremental.seed(&assignment).unwrap();

        assert!(incremental.update_delta(&assignment, &[(0, 1, 0)]).is_err());
        assert!(incremental.update_delta(&assignment, &[(6, 0, 1)]).is_err());
    }
}
