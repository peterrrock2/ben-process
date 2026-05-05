use crate::graph::Graph;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline};
use polars::prelude::*;
use std::fs::File;
use std::io;

/// Upper bound on district ids the dense-buffer path supports.
/// `observed` is a u128 bitmask indexed by district id.
const MAX_DISTRICTS: u16 = 128;

/// Per-sample tally result.
///
/// `totals` is a flat `Vec<f64>` of shape `[n_keys * n_districts]`, where
/// `n_districts = max(assignment) + 1`. `observed` has bit `d` set iff
/// district `d` appeared in this sample's assignment. Keeping `observed`
/// lets the writer emit columns only for districts that actually show up
/// somewhere in the ensemble (matching the pre-Phase-3 output shape).
struct TallyRow {
    sample_num: u64,
    n_reps: u32,
    accepted_count: u32,
    totals: Vec<f64>,
    n_districts: u16,
    observed: u128,
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

fn save_tallies_to_parquet(
    file_path: &str,
    tallies: &[TallyRow],
    key_list: &[String],
    high_compression: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Global union of districts observed anywhere in the ensemble. Columns
    // come out in ascending district-id order, same as the BTreeMap fix
    // from Phase 0.
    let global_observed: u128 = tallies.iter().fold(0u128, |acc, t| acc | t.observed);
    let district_ids = sorted_district_ids(global_observed);

    let n_rows = tallies.len() * key_list.len();
    let mut sample_numbers: Vec<u64> = Vec::with_capacity(n_rows);
    let mut n_reps_numbers: Vec<u32> = Vec::with_capacity(n_rows);
    let mut accepted_numbers: Vec<u32> = Vec::with_capacity(n_rows);
    let mut sum_columns: Vec<String> = Vec::with_capacity(n_rows);
    let mut district_cols: Vec<Vec<Option<f64>>> = district_ids
        .iter()
        .map(|_| Vec::with_capacity(n_rows))
        .collect();

    for row in tallies {
        let n_d = row.n_districts as usize;
        for (k, key) in key_list.iter().enumerate() {
            sample_numbers.push(row.sample_num);
            n_reps_numbers.push(row.n_reps);
            accepted_numbers.push(row.accepted_count);
            sum_columns.push(key.clone());

            let offset = k * n_d;
            for (ci, &d) in district_ids.iter().enumerate() {
                let di = d as usize;
                let present = di < n_d && (row.observed & (1u128 << d)) != 0;
                district_cols[ci].push(if present { Some(row.totals[offset + di]) } else { None });
            }
        }
    }

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_numbers).into(),
        Series::new("n_reps".into(), n_reps_numbers).into(),
        Series::new("accepted_count".into(), accepted_numbers).into(),
        Series::new("sum_columns".into(), sum_columns).into(),
    ])?;

    for (ci, &d) in district_ids.iter().enumerate() {
        let col = std::mem::take(&mut district_cols[ci]);
        df.with_column(Series::new(format!("district_{}", d).into(), col).into())?;
    }

    let mut file = File::create(file_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(parquet_compression(high_compression))
        .finish(&mut df)?;

    Ok(())
}

pub fn tally_and_save_from_key_list(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    key_list: Vec<String>,
    show_progress: bool,
    high_compression: bool,
) -> io::Result<()> {
    let attr_col_indices: Vec<usize> = key_list
        .iter()
        .map(|k| {
            *graph
                .attr_index
                .get(k)
                .unwrap_or_else(|| panic!("key {:?} not pre-loaded on graph", k))
        })
        .collect();

    let total = count_samples(in_file_name)?;
    let mut all_tallies: Vec<TallyRow> = Vec::with_capacity(total);

    run_pipeline(
        in_file_name,
        total,
        |assignment, _n_reps| tally_keys(&graph, assignment, &attr_col_indices),
        |step, n_reps, accepted, row| {
            let (totals, n_districts, observed) = row;
            all_tallies.push(TallyRow {
                sample_num: step,
                n_reps,
                accepted_count: accepted,
                totals,
                n_districts,
                observed,
            });
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    save_tallies_to_parquet(out_file_name, &all_tallies, &key_list, high_compression)
        .expect("Unable to save tallies");
    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{save_tallies_to_parquet, sorted_district_ids, tally_keys, TallyRow};
    use crate::graph::Graph;
    use polars::prelude::{ParquetReader, SerReader};
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
    fn save_tallies_to_parquet_writes_union_of_observed_districts_with_nulls() {
        let file = NamedTempFile::new().unwrap();
        let key_list = vec!["pop".to_string()];
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

        save_tallies_to_parquet(file.path().to_str().unwrap(), &tallies, &key_list, false)
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
                "sum_columns",
                "district_1",
                "district_3",
            ]
        );
        assert_eq!(district_1.get(0), Some(60.0));
        assert_eq!(district_1.get(1), None);
        assert_eq!(district_3.get(0), None);
        assert_eq!(district_3.get(1), Some(20.0));
    }
}
