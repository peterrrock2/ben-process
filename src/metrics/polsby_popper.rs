use crate::district::observed_assignment_districts;
use crate::graph::Graph;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
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
    let (n_districts, observed) = observed_assignment_districts(assignment)?;
    let n_districts = n_districts as usize;
    let mut area_by_district = vec![0.0f64; n_districts];
    let mut perimeter_by_district = vec![0.0f64; n_districts];

    for (node, &district) in assignment.iter().enumerate() {
        let district = district as usize;
        area_by_district[district] += area_values[node];
        perimeter_by_district[district] += total_perimeter_values[node];
    }

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

#[allow(clippy::too_many_arguments)]
pub fn tally_and_save_polsby_popper(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    area_key: &str,
    perim_key: Option<&str>,
    boundary_perim_key: Option<&str>,
    show_progress: bool,
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

    run_pipeline(
        in_file_name,
        Some(graph.node_count),
        // The pipeline enforces a fixed district set, so the schema fixed from the first row
        // holds.
        Some("polsby-popper"),
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
            writer.push_row(step, n_reps, accepted, observed, &scores)
        },
        show_progress,
    )?;

    log::info!("Writing final output...");
    writer.finish()?;
    log::info!("Done!");
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
}
