use crate::cli::{build_tally_output_dir, build_tally_output_path};
use crate::district::{observed_assignment_districts, sorted_district_ids};
use crate::graph::Graph;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use std::fs::{create_dir_all, File};
use std::io;

/// Per-sample tally result.
///
/// `totals` is a flat `Vec<f64>` of shape `[n_keys * n_districts]`, where
/// `n_districts = max(assignment) + 1`. `observed` has bit `d` set iff district `d` appeared in
/// this sample's assignment. Keeping `observed` lets the writer emit columns only for districts
/// that actually show up in the first assignment, which fixes the per-key parquet schemas.
struct TallyRow {
    sample_number: u64,
    n_reps: u32,
    accepted_count: u64,
    totals: Vec<f64>,
    n_districts: u16,
    observed: u128,
}

/// Hot loop: flat index into pre-parsed attribute columns, accumulate into a flat per-district
/// totals vector. No HashMap work inside the inner loop.
fn tally_keys(
    graph: &Graph,
    assignment: &[u16],
    attr_column_indices: &[usize],
) -> io::Result<(Vec<f64>, u16, u128)> {
    // The assignment is guaranteed to have one entry per graph node by `run_pipeline`'s length
    // check; this hot loop relies on that invariant when indexing `assignment[node_index]` below.
    let (n_districts, observed) = observed_assignment_districts(assignment)?;
    let n_districts = n_districts as usize;
    let n_keys = attr_column_indices.len();
    let mut totals = vec![0.0f64; n_keys * n_districts];
    for (key_index, &column_index) in attr_column_indices.iter().enumerate() {
        let column = &graph.attr_columns[column_index];
        let offset = key_index * n_districts;
        for (node_index, &value) in column.iter().enumerate() {
            totals[offset + assignment[node_index] as usize] += value;
        }
    }
    Ok((totals, n_districts as u16, observed))
}

fn make_key_writer_state(
    ben_file_name: &str,
    output_dir: Option<&str>,
    key: &str,
    district_ids: &[u16],
    high_compression: bool,
) -> Result<DistrictMetricWriter, Box<dyn std::error::Error>> {
    let output_path = build_tally_output_path(ben_file_name, key, output_dir);
    let file = File::create(output_path)?;
    DistrictMetricWriter::new(
        file,
        district_ids.to_vec(),
        parquet_compression(high_compression),
        PARQUET_BATCH_ROWS,
    )
}

pub fn tally_and_save_from_key_list(
    graph: Graph,
    in_file_name: &str,
    output_dir: Option<&str>,
    key_list: Vec<String>,
    show_progress: bool,
    high_compression: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let attr_column_indices: Vec<usize> = key_list
        .iter()
        .map(|key| {
            graph
                .numeric_column_index(key)
                .unwrap_or_else(|| panic!("key {:?} not pre-loaded on graph", key))
        })
        .collect();

    create_dir_all(build_tally_output_dir(in_file_name, output_dir))?;

    let mut key_states: Option<Vec<DistrictMetricWriter>> = None;
    let mut district_ids: Vec<u16> = Vec::new();

    run_pipeline(
        in_file_name,
        Some(graph.node_count),
        // The pipeline enforces that this district set is identical for every plan, so the schema
        // fixed from the first row below holds for the whole run.
        Some("tally"),
        |assignment, _n_reps| {
            let (totals, n_districts, observed) =
                tally_keys(&graph, assignment, &attr_column_indices)?;
            Ok((observed, (totals, n_districts, observed)))
        },
        |step, n_reps, accepted, row| {
            let (totals, n_districts, observed) = row;
            let row = TallyRow {
                sample_number: step,
                n_reps,
                accepted_count: accepted,
                totals,
                n_districts,
                observed,
            };

            if key_states.is_none() {
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
                        .map_err(|e| io::Error::other(e.to_string()))?,
                );
            }

            for (key_index, state) in key_states
                .as_mut()
                .expect("writers should exist before writing tallies")
                .iter_mut()
                .enumerate()
            {
                let n_districts = row.n_districts as usize;
                let offset = key_index * n_districts;
                state
                    .push_row_with(
                        row.sample_number,
                        row.n_reps,
                        row.accepted_count,
                        |district| {
                            let district_index = district as usize;
                            let present = district_index < n_districts
                                && (row.observed & (1u128 << district)) != 0;
                            if present {
                                Some(row.totals[offset + district_index])
                            } else {
                                None
                            }
                        },
                    )
                    .map_err(|e| io::Error::other(e.to_string()))?;
            }
            Ok(())
        },
        show_progress,
    )?;

    log::info!("Writing final output...");
    let key_states = match key_states {
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

    for state in key_states {
        state.finish()?;
    }
    log::info!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::tally_keys;
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
        }
    }

    #[test]
    fn tally_keys_accumulates_multiple_keys_and_sparse_district_ids() {
        let graph = graph_with_attr_columns(vec![vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]]);

        let (totals, n_districts, observed) = tally_keys(&graph, &[1, 3, 1], &[0, 1]).unwrap();

        assert_eq!(n_districts, 4);
        assert_eq!(observed, (1u128 << 1) | (1u128 << 3));
        assert_eq!(totals, vec![0.0, 4.0, 0.0, 2.0, 0.0, 40.0, 0.0, 20.0]);
    }
}
