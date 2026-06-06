use crate::district::{
    assert_no_unseen_districts, observed_assignment_districts, sorted_district_ids,
};
use crate::graph::Graph;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use std::fs::File;

struct PolsbyRow {
    sample_num: u64,
    n_reps: u32,
    accepted_count: u32,
    scores: Vec<f64>,
    n_districts: u16,
    observed: u128,
}

#[inline]
fn polsby_popper_score(area: f64, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        0.0
    } else {
        4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
    }
}

fn derive_total_perimeters(
    boundary_perims: &[f64],
    edges: &[(u32, u32)],
    shared_perims: &[f64],
) -> Vec<f64> {
    let mut total_perims = boundary_perims.to_vec();
    for (i, &(u, v)) in edges.iter().enumerate() {
        total_perims[u as usize] += shared_perims[i];
        total_perims[v as usize] += shared_perims[i];
    }
    total_perims
}

fn polsby_popper_rows(
    assignment: &[u16],
    area_vals: &[f64],
    total_perim_vals: &[f64],
    edges: &[(u32, u32)],
    shared_perims: &[f64],
) -> (Vec<f64>, u16, u128) {
    let (n_districts, observed) = observed_assignment_districts(assignment);
    let n_districts = n_districts as usize;
    let mut area_d = vec![0.0f64; n_districts];
    let mut perim_d = vec![0.0f64; n_districts];

    for (node, &district) in assignment.iter().enumerate() {
        let district = district as usize;
        area_d[district] += area_vals[node];
        perim_d[district] += total_perim_vals[node];
    }

    for (edge_idx, &(u, v)) in edges.iter().enumerate() {
        let d_u = assignment[u as usize] as usize;
        let d_v = assignment[v as usize] as usize;
        if d_u == d_v {
            perim_d[d_u] -= 2.0 * shared_perims[edge_idx];
        }
    }

    let scores = (0..n_districts)
        .map(|d| polsby_popper_score(area_d[d], perim_d[d]))
        .collect();
    (scores, n_districts as u16, observed)
}

#[allow(clippy::too_many_arguments)]
pub fn tally_and_save_polsby_popper(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    area_key: &str,
    perim_key: Option<&str>,
    boundary_perim_key: Option<&str>,
    _shared_perim_key: &str,
    show_progress: bool,
    high_compression: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let area_vals = graph
        .numeric_column(area_key)
        .unwrap_or_else(|| panic!("area key {:?} not pre-loaded on graph", area_key));

    let shared_perims = graph
        .edge_weight_column()
        .unwrap_or_else(|| panic!("shared perimeter edge column not pre-loaded on graph"));

    let total_perims = if let Some(perim_key) = perim_key {
        graph
            .numeric_column(perim_key)
            .unwrap_or_else(|| panic!("perimeter key {:?} not pre-loaded on graph", perim_key))
            .to_vec()
    } else {
        let boundary_key = boundary_perim_key
            .expect("boundary perimeter key should exist when direct perimeter key is absent");
        let boundary_perims = graph.numeric_column(boundary_key).unwrap_or_else(|| {
            panic!(
                "boundary perimeter key {:?} not pre-loaded on graph",
                boundary_key
            )
        });
        derive_total_perimeters(boundary_perims, &graph.edges, shared_perims)
    };

    let total = count_samples(in_file_name)?;
    let mut file = Some(File::create(out_file_name)?);
    let mut writer_state: Option<DistrictMetricWriter> = None;
    let mut expected_observed: Option<u128> = None;

    run_pipeline(
        in_file_name,
        total,
        Some(graph.node_count),
        |assignment, _n_reps| {
            polsby_popper_rows(
                assignment,
                area_vals,
                &total_perims,
                &graph.edges,
                shared_perims,
            )
        },
        |step, n_reps, accepted, row| {
            let (scores, n_districts, observed) = row;
            let row = PolsbyRow {
                sample_num: step,
                n_reps,
                accepted_count: accepted,
                scores,
                n_districts,
                observed,
            };

            if writer_state.is_none() {
                expected_observed = Some(row.observed);
                let district_ids = sorted_district_ids(row.observed);
                writer_state = Some(
                    DistrictMetricWriter::new(
                        file.take()
                            .expect("output file should be available when initializing writer"),
                        district_ids.clone(),
                        parquet_compression(high_compression),
                        PARQUET_BATCH_ROWS,
                    )
                    .expect("Unable to initialize polsby-popper parquet writer"),
                );
            }

            let expected = expected_observed.expect("writer should initialize on first row");
            assert_no_unseen_districts(row.observed, expected, "polsby-popper");

            let state = writer_state
                .as_mut()
                .expect("writer should exist before writing polsby-popper rows");
            let n_d = row.n_districts as usize;
            state
                .push_row_with(row.sample_num, row.n_reps, row.accepted_count, |d| {
                    let di = d as usize;
                    let present = di < n_d && (row.observed & (1u128 << d)) != 0;
                    if present {
                        Some(row.scores[di])
                    } else {
                        None
                    }
                })
                .expect("Unable to write polsby-popper row");
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    let writer_state = match writer_state {
        Some(state) => state,
        None => DistrictMetricWriter::new(
            file.take()
                .expect("output file should be available when initializing empty writer"),
            vec![],
            parquet_compression(high_compression),
            PARQUET_BATCH_ROWS,
        )?,
    };
    writer_state.finish()?;
    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{derive_total_perimeters, polsby_popper_rows, polsby_popper_score};

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
    fn polsby_popper_rows_matches_known_two_district_example() {
        let (scores, n_districts, observed) = polsby_popper_rows(
            &[1, 1, 2, 2],
            &[1.0, 1.0, 1.0, 1.0],
            &[4.0, 4.0, 4.0, 4.0],
            &[(0, 1), (1, 2), (2, 3)],
            &[1.0, 1.0, 1.0],
        );

        let expected = 2.0 * std::f64::consts::PI / 9.0;
        assert_eq!(n_districts, 3);
        assert_eq!(observed, (1u128 << 1) | (1u128 << 2));
        assert_eq!(scores[0], 0.0);
        assert!((scores[1] - expected).abs() < 1e-12);
        assert!((scores[2] - expected).abs() < 1e-12);
    }
}
