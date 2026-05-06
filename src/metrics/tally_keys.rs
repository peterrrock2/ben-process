use crate::cli::{build_tally_output_dir, build_tally_output_path};
use crate::graph::Graph;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use polars::io::parquet::write::BatchedWriter;
use polars::prelude::*;
use std::fs::{create_dir_all, File};

/// Upper bound on district ids the dense-buffer path supports.
/// `observed` is a u128 bitmask indexed by district id.
const MAX_DISTRICTS: u16 = 128;

/// Per-sample tally result.
///
/// `totals` is a flat `Vec<f64>` of shape `[n_keys * n_districts]`, where
/// `n_districts = max(assignment) + 1`. `observed` has bit `d` set iff
/// district `d` appeared in this sample's assignment. Keeping `observed`
/// lets the writer emit columns only for districts that actually show up
/// in the first assignment, which fixes the per-key parquet schemas.
struct TallyRow {
    sample_num: u64,
    n_reps: u32,
    accepted_count: u32,
    totals: Vec<f64>,
    n_districts: u16,
    observed: u128,
}

struct KeyWriterState {
    writer: BatchedWriter<File>,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u32>,
    district_cols: Vec<Vec<Option<f64>>>,
}

/// Hot loop: flat index into pre-parsed attribute columns, accumulate into a
/// flat per-district totals vector. No HashMap work inside the inner loop.
fn tally_keys(
    graph: &Graph,
    assignment: &[u16],
    attr_col_indices: &[usize],
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
    let n_keys = attr_col_indices.len();
    let mut totals = vec![0.0f64; n_keys * n_districts];
    for (k, &col_idx) in attr_col_indices.iter().enumerate() {
        let col = &graph.attr_columns[col_idx];
        let offset = k * n_districts;
        for (i, &v) in col.iter().enumerate() {
            totals[offset + assignment[i] as usize] += v;
        }
    }
    (totals, n_districts as u16, observed)
}

/// Bits set in `mask`, returned in ascending order.
fn sorted_district_ids(mut mask: u128) -> Vec<u16> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        out.push(mask.trailing_zeros() as u16);
        mask &= mask - 1;
    }
    out
}

fn empty_key_df(district_ids: &[u16]) -> PolarsResult<DataFrame> {
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

fn key_batch_to_df(
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

fn make_key_writer_state(
    ben_file_name: &str,
    output_dir: Option<&str>,
    key: &str,
    district_ids: &[u16],
    high_compression: bool,
) -> Result<KeyWriterState, Box<dyn std::error::Error>> {
    let output_path = build_tally_output_path(ben_file_name, key, output_dir);
    let file = File::create(output_path)?;
    let empty_df = empty_key_df(district_ids)?;
    let writer = ParquetWriter::new(file)
        .with_compression(parquet_compression(high_compression))
        .batched(empty_df.schema())?;

    Ok(KeyWriterState {
        writer,
        sample_numbers: Vec::with_capacity(PARQUET_BATCH_ROWS),
        n_reps_numbers: Vec::with_capacity(PARQUET_BATCH_ROWS),
        accepted_numbers: Vec::with_capacity(PARQUET_BATCH_ROWS),
        district_cols: district_ids
            .iter()
            .map(|_| Vec::with_capacity(PARQUET_BATCH_ROWS))
            .collect(),
    })
}

fn flush_key_writer(
    district_ids: &[u16],
    state: &mut KeyWriterState,
) -> Result<(), Box<dyn std::error::Error>> {
    if state.sample_numbers.is_empty() {
        return Ok(());
    }

    let df = key_batch_to_df(
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

#[cfg(test)]
fn save_single_key_tallies_to_parquet(
    file_path: &str,
    tallies: &[TallyRow],
    key_index: usize,
    high_compression: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let global_observed: u128 = tallies.iter().fold(0u128, |acc, t| acc | t.observed);
    let district_ids = sorted_district_ids(global_observed);

    let mut sample_numbers: Vec<u64> = Vec::with_capacity(tallies.len());
    let mut n_reps_numbers: Vec<u32> = Vec::with_capacity(tallies.len());
    let mut accepted_numbers: Vec<u32> = Vec::with_capacity(tallies.len());
    let mut district_cols: Vec<Vec<Option<f64>>> = district_ids
        .iter()
        .map(|_| Vec::with_capacity(tallies.len()))
        .collect();

    for row in tallies {
        sample_numbers.push(row.sample_num);
        n_reps_numbers.push(row.n_reps);
        accepted_numbers.push(row.accepted_count);

        let n_d = row.n_districts as usize;
        let offset = key_index * n_d;
        for (ci, &d) in district_ids.iter().enumerate() {
            let di = d as usize;
            let present = di < n_d && (row.observed & (1u128 << d)) != 0;
            district_cols[ci].push(if present {
                Some(row.totals[offset + di])
            } else {
                None
            });
        }
    }

    let mut df = key_batch_to_df(
        &district_ids,
        &mut sample_numbers,
        &mut n_reps_numbers,
        &mut accepted_numbers,
        &mut district_cols,
    )?;
    let mut file = File::create(file_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(parquet_compression(high_compression))
        .finish(&mut df)?;

    Ok(())
}

pub fn tally_and_save_from_key_list(
    graph: Graph,
    in_file_name: &str,
    output_dir: Option<&str>,
    key_list: Vec<String>,
    show_progress: bool,
    high_compression: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let attr_col_indices: Vec<usize> = key_list
        .iter()
        .map(|k| {
            *graph
                .attr_index
                .get(k)
                .unwrap_or_else(|| panic!("key {:?} not pre-loaded on graph", k))
        })
        .collect();

    create_dir_all(build_tally_output_dir(in_file_name, output_dir))?;

    let total = count_samples(in_file_name)?;
    let mut key_states: Option<Vec<KeyWriterState>> = None;
    let mut district_ids: Vec<u16> = Vec::new();
    let mut expected_observed: Option<u128> = None;

    run_pipeline(
        in_file_name,
        total,
        |assignment, _n_reps| tally_keys(&graph, assignment, &attr_col_indices),
        |step, n_reps, accepted, row| {
            let (totals, n_districts, observed) = row;
            let row = TallyRow {
                sample_num: step,
                n_reps,
                accepted_count: accepted,
                totals,
                n_districts,
                observed,
            };

            if key_states.is_none() {
                expected_observed = Some(row.observed);
                district_ids = sorted_district_ids(row.observed);
                key_states = Some(
                    key_list
                        .iter()
                        .map(|key| {
                            make_key_writer_state(
                                in_file_name,
                                output_dir,
                                key,
                                &district_ids,
                                high_compression,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .expect("Unable to initialize per-key tally parquet writers"),
                );
            }

            let expected = expected_observed.expect("writers should initialize on first row");
            let unseen = row.observed & !expected;
            if unseen != 0 {
                panic!(
                    "encountered districts {:?} not present in first assignment; cannot stream tally output with a fixed schema",
                    sorted_district_ids(unseen)
                );
            }

            for (key_idx, state) in key_states
                .as_mut()
                .expect("writers should exist before writing tallies")
                .iter_mut()
                .enumerate()
            {
                state.sample_numbers.push(row.sample_num);
                state.n_reps_numbers.push(row.n_reps);
                state.accepted_numbers.push(row.accepted_count);

                let n_d = row.n_districts as usize;
                let offset = key_idx * n_d;
                for (ci, &d) in district_ids.iter().enumerate() {
                    let di = d as usize;
                    let present = di < n_d && (row.observed & (1u128 << d)) != 0;
                    state.district_cols[ci].push(if present {
                        Some(row.totals[offset + di])
                    } else {
                        None
                    });
                }

                if state.sample_numbers.len() >= PARQUET_BATCH_ROWS {
                    flush_key_writer(&district_ids, state)
                        .expect("Unable to flush per-key tally batch");
                }
            }
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    let mut key_states = match key_states {
        Some(states) => states,
        None => {
            district_ids = vec![];
            key_list
                .iter()
                .map(|key| {
                    make_key_writer_state(
                        in_file_name,
                        output_dir,
                        key,
                        &district_ids,
                        high_compression,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    };

    for state in &mut key_states {
        flush_key_writer(&district_ids, state)?;
        state.writer.finish()?;
    }
    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        empty_key_df, key_batch_to_df, save_single_key_tallies_to_parquet, sorted_district_ids,
        tally_keys, TallyRow,
    };
    use crate::graph::Graph;
    use crate::pipeline::parquet_compression;
    use polars::prelude::{ParquetReader, ParquetWriter, SerReader};
    use std::collections::HashMap;
    use std::fs::File;
    use tempfile::NamedTempFile;

    fn graph_with_attr_columns(attr_columns: Vec<Vec<f64>>) -> Graph {
        Graph {
            attr_columns,
            attr_index: HashMap::new(),
            region_columns: vec![],
            region_index: HashMap::new(),
            region_id_counts: vec![],
            edges: vec![],
            edge_weights: None,
        }
    }

    #[test]
    fn tally_keys_accumulates_multiple_keys_and_sparse_district_ids() {
        let graph = graph_with_attr_columns(vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]);

        let (totals, n_districts, observed) = tally_keys(&graph, &[1, 3, 1], &[0, 1]);

        assert_eq!(n_districts, 4);
        assert_eq!(observed, (1u128 << 1) | (1u128 << 3));
        assert_eq!(totals, vec![0.0, 4.0, 0.0, 2.0, 0.0, 40.0, 0.0, 20.0]);
    }

    #[test]
    #[should_panic(expected = "district id 128 exceeds current 128-district limit")]
    fn tally_keys_panics_when_assignment_exceeds_supported_district_limit() {
        let graph = graph_with_attr_columns(vec![vec![1.0]]);
        let _ = tally_keys(&graph, &[128], &[0]);
    }

    #[test]
    fn sorted_district_ids_returns_ascending_order() {
        let mask = (1u128 << 63) | (1u128 << 1) | (1u128 << 65);
        assert_eq!(sorted_district_ids(mask), vec![1, 63, 65]);
    }

    #[test]
    fn save_single_key_tallies_to_parquet_writes_union_of_observed_districts_with_nulls() {
        let file = NamedTempFile::new().unwrap();
        let tallies = vec![
            TallyRow {
                sample_num: 1,
                n_reps: 1,
                accepted_count: 1,
                totals: vec![0.0, 60.0],
                n_districts: 2,
                observed: 1u128 << 1,
            },
            TallyRow {
                sample_num: 2,
                n_reps: 1,
                accepted_count: 2,
                totals: vec![0.0, 0.0, 0.0, 20.0],
                n_districts: 4,
                observed: 1u128 << 3,
            },
        ];

        save_single_key_tallies_to_parquet(file.path().to_str().unwrap(), &tallies, 0, false)
            .unwrap();

        let df = ParquetReader::new(&mut File::open(file.path()).unwrap())
            .finish()
            .unwrap();
        let district_1 = df.column("district_1").unwrap().f64().unwrap();
        let district_3 = df.column("district_3").unwrap().f64().unwrap();

        assert_eq!(
            df.get_column_names()
                .iter()
                .map(|name| name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "step",
                "n_reps",
                "accepted_count",
                "district_1",
                "district_3",
            ]
        );
        assert_eq!(district_1.get(0), Some(60.0));
        assert_eq!(district_1.get(1), None);
        assert_eq!(district_3.get(0), None);
        assert_eq!(district_3.get(1), Some(20.0));
    }

    #[test]
    fn key_batched_writer_appends_multiple_batches() {
        let file = NamedTempFile::new().unwrap();
        let district_ids = vec![1, 2];
        let empty_df = empty_key_df(&district_ids).unwrap();
        let mut writer = ParquetWriter::new(File::create(file.path()).unwrap())
            .with_compression(parquet_compression(false))
            .batched(empty_df.schema())
            .unwrap();

        let mut batch1_steps = vec![1, 2];
        let mut batch1_reps = vec![1, 1];
        let mut batch1_accepted = vec![1, 2];
        let mut batch1_cols = vec![vec![Some(10.0), Some(20.0)], vec![Some(30.0), Some(40.0)]];
        let batch1 = key_batch_to_df(
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
        let mut batch2_cols = vec![vec![Some(50.0)], vec![None]];
        let batch2 = key_batch_to_df(
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
            vec![Some(10.0), Some(20.0), Some(50.0)]
        );
        assert_eq!(
            df.column("district_2")
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![Some(30.0), Some(40.0), None]
        );
    }
}
