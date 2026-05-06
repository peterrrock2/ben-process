use crate::graph::Graph;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use polars::prelude::*;
use std::fs::File;

const MAX_DISTRICTS: u16 = 128;

struct PolsbyRow {
    sample_num: u64,
    n_reps: u32,
    accepted_count: u32,
    scores: Vec<f64>,
    n_districts: u16,
    observed: u128,
}

struct PolsbyWriterState {
    writer: polars::io::parquet::write::BatchedWriter<File>,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u32>,
    district_cols: Vec<Vec<Option<f64>>>,
}

#[inline]
fn polsby_popper_score(area: f64, perimeter: f64) -> f64 {
    if perimeter <= 0.0 {
        0.0
    } else {
        4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
    }
}

fn sorted_district_ids(mut mask: u128) -> Vec<u16> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        out.push(mask.trailing_zeros() as u16);
        mask &= mask - 1;
    }
    out
}

fn derive_total_perimeters(boundary_perims: &[f64], edges: &[(u32, u32)], shared_perims: &[f64]) -> Vec<f64> {
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
    let mut observed: u128 = 0;
    let mut max_d: u16 = 0;
    for &d in assignment {
        if d >= MAX_DISTRICTS {
            panic!(
                "district id {} exceeds current {}-district limit; widen the observed bitmask",
                d, MAX_DISTRICTS
            );
        }
        observed |= 1u128 << d;
        if d > max_d {
            max_d = d;
        }
    }

    let n_districts = max_d as usize + 1;
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

fn empty_polsby_df(district_ids: &[u16]) -> PolarsResult<DataFrame> {
    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), Vec::<u64>::new()).into(),
        Series::new("n_reps".into(), Vec::<u32>::new()).into(),
        Series::new("accepted_count".into(), Vec::<u32>::new()).into(),
    ])?;

    for &d in district_ids {
        df.with_column(
            Series::new(format!("district_{}", d).into(), Vec::<Option<f64>>::new()).into(),
        )?;
    }

    Ok(df)
}

fn polsby_batch_to_df(
    district_ids: &[u16],
    sample_numbers: &mut Vec<u64>,
    n_reps_numbers: &mut Vec<u32>,
    accepted_numbers: &mut Vec<u32>,
    district_cols: &mut Vec<Vec<Option<f64>>>,
) -> PolarsResult<DataFrame> {
    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), std::mem::take(sample_numbers)).into(),
        Series::new("n_reps".into(), std::mem::take(n_reps_numbers)).into(),
        Series::new("accepted_count".into(), std::mem::take(accepted_numbers)).into(),
    ])?;

    for (ci, &d) in district_ids.iter().enumerate() {
        let col = std::mem::take(&mut district_cols[ci]);
        df.with_column(Series::new(format!("district_{}", d).into(), col).into())?;
    }

    Ok(df)
}

fn flush_writer(
    district_ids: &[u16],
    state: &mut PolsbyWriterState,
) -> Result<(), Box<dyn std::error::Error>> {
    if state.sample_numbers.is_empty() {
        return Ok(());
    }

    let df = polsby_batch_to_df(
        district_ids,
        &mut state.sample_numbers,
        &mut state.n_reps_numbers,
        &mut state.accepted_numbers,
        &mut state.district_cols,
    )?;
    state.writer.write_batch(&df)?;
    state.district_cols = district_ids
        .iter()
        .map(|_| Vec::with_capacity(PARQUET_BATCH_ROWS))
        .collect();

    Ok(())
}

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
    let area_idx = *graph
        .attr_index
        .get(area_key)
        .unwrap_or_else(|| panic!("area key {:?} not pre-loaded on graph", area_key));
    let area_vals = &graph.attr_columns[area_idx];

    let shared_perims = graph
        .edge_weights
        .as_ref()
        .unwrap_or_else(|| panic!("shared perimeter edge column not pre-loaded on graph"));

    let total_perims = if let Some(perim_key) = perim_key {
        let perim_idx = *graph
            .attr_index
            .get(perim_key)
            .unwrap_or_else(|| panic!("perimeter key {:?} not pre-loaded on graph", perim_key));
        graph.attr_columns[perim_idx].clone()
    } else {
        let boundary_key = boundary_perim_key.expect(
            "boundary perimeter key should exist when direct perimeter key is absent",
        );
        let boundary_idx = *graph.attr_index.get(boundary_key).unwrap_or_else(|| {
            panic!(
                "boundary perimeter key {:?} not pre-loaded on graph",
                boundary_key
            )
        });
        derive_total_perimeters(&graph.attr_columns[boundary_idx], &graph.edges, shared_perims)
    };

    let total = count_samples(in_file_name)?;
    let mut file = Some(File::create(out_file_name)?);
    let mut writer_state: Option<PolsbyWriterState> = None;
    let mut district_ids: Vec<u16> = Vec::new();
    let mut expected_observed: Option<u128> = None;

    run_pipeline(
        in_file_name,
        total,
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
                district_ids = sorted_district_ids(row.observed);
                let empty_df = empty_polsby_df(&district_ids)
                    .expect("Unable to build polsby-popper schema DataFrame");
                let writer = ParquetWriter::new(
                    file.take()
                        .expect("output file should be available when initializing writer"),
                )
                .with_compression(parquet_compression(high_compression))
                .batched(empty_df.schema())
                .expect("Unable to initialize polsby-popper parquet writer");
                writer_state = Some(PolsbyWriterState {
                    writer,
                    sample_numbers: Vec::with_capacity(PARQUET_BATCH_ROWS),
                    n_reps_numbers: Vec::with_capacity(PARQUET_BATCH_ROWS),
                    accepted_numbers: Vec::with_capacity(PARQUET_BATCH_ROWS),
                    district_cols: district_ids
                        .iter()
                        .map(|_| Vec::with_capacity(PARQUET_BATCH_ROWS))
                        .collect(),
                });
            }

            let expected = expected_observed.expect("writer should initialize on first row");
            let unseen = row.observed & !expected;
            if unseen != 0 {
                panic!(
                    "encountered districts {:?} not present in first assignment; cannot stream polsby-popper output with a fixed schema",
                    sorted_district_ids(unseen)
                );
            }

            let state = writer_state
                .as_mut()
                .expect("writer should exist before writing polsby-popper rows");
            state.sample_numbers.push(row.sample_num);
            state.n_reps_numbers.push(row.n_reps);
            state.accepted_numbers.push(row.accepted_count);

            let n_d = row.n_districts as usize;
            for (ci, &d) in district_ids.iter().enumerate() {
                let di = d as usize;
                let present = di < n_d && (row.observed & (1u128 << d)) != 0;
                state.district_cols[ci].push(if present {
                    Some(row.scores[di])
                } else {
                    None
                });
            }

            if state.sample_numbers.len() >= PARQUET_BATCH_ROWS {
                flush_writer(&district_ids, state)
                    .expect("Unable to flush polsby-popper batch");
            }
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    let mut writer_state = match writer_state {
        Some(state) => state,
        None => {
            let empty_df = empty_polsby_df(&[])?;
            let writer = ParquetWriter::new(
                file.take()
                    .expect("output file should be available when initializing empty writer"),
            )
            .with_compression(parquet_compression(high_compression))
            .batched(empty_df.schema())?;
            PolsbyWriterState {
                writer,
                sample_numbers: Vec::new(),
                n_reps_numbers: Vec::new(),
                accepted_numbers: Vec::new(),
                district_cols: Vec::new(),
            }
        }
    };
    flush_writer(&district_ids, &mut writer_state)?;
    writer_state.writer.finish()?;
    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{derive_total_perimeters, polsby_popper_rows, polsby_popper_score, polsby_batch_to_df};
    use crate::pipeline::parquet_compression;
    use polars::prelude::{ParquetReader, ParquetWriter, SerReader};
    use std::fs::File;
    use tempfile::NamedTempFile;

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

    #[test]
    fn polsby_batched_writer_appends_multiple_batches() {
        let file = NamedTempFile::new().unwrap();
        let district_ids = vec![1, 2];
        let empty_df = super::empty_polsby_df(&district_ids).unwrap();
        let mut writer = ParquetWriter::new(File::create(file.path()).unwrap())
            .with_compression(parquet_compression(false))
            .batched(empty_df.schema())
            .unwrap();

        let mut batch1_steps = vec![1, 2];
        let mut batch1_reps = vec![1, 1];
        let mut batch1_accepted = vec![1, 2];
        let mut batch1_cols = vec![vec![Some(0.1), Some(0.2)], vec![Some(0.3), Some(0.4)]];
        let batch1 = polsby_batch_to_df(
            &district_ids,
            &mut batch1_steps,
            &mut batch1_reps,
            &mut batch1_accepted,
            &mut batch1_cols,
        )
        .unwrap();
        writer.write_batch(&batch1).unwrap();

        let mut batch2_steps = vec![3];
        let mut batch2_reps = vec![2];
        let mut batch2_accepted = vec![3];
        let mut batch2_cols = vec![vec![Some(0.5)], vec![None]];
        let batch2 = polsby_batch_to_df(
            &district_ids,
            &mut batch2_steps,
            &mut batch2_reps,
            &mut batch2_accepted,
            &mut batch2_cols,
        )
        .unwrap();
        writer.write_batch(&batch2).unwrap();
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
            df.column("district_1")
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![Some(0.1), Some(0.2), Some(0.5)]
        );
    }
}
