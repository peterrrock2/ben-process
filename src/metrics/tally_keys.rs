use crate::cli::{build_tally_output_dir, build_tally_output_path};
use crate::district::observed_assignment_districts;
use crate::graph::Graph;
use crate::input::BenSource;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{
    parquet_compression, run_pipeline, AssignmentLengthCheck, PARQUET_BATCH_ROWS,
};
use std::fs::{create_dir_all, File};

/// Hot loop: flat index into pre-parsed attribute columns, accumulate into a flat per-district
/// totals vector. No HashMap work inside the inner loop.
///
/// `totals` is a flat `Vec<f64>` of shape `[n_keys * n_districts]`, where
/// `n_districts = max(assignment) + 1`. `observed` has bit `d` set iff district `d` appeared in
/// this sample's assignment.
fn tally_keys(
    graph: &Graph,
    assignment: &[u16],
    attr_column_indices: &[usize],
) -> crate::error::Result<(Vec<f64>, u16, u128)> {
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

pub fn tally_and_save_from_key_list(
    graph: Graph,
    source: &BenSource,
    output_dir: Option<&str>,
    key_list: Vec<String>,
    show_progress: bool,
    max_samples: Option<usize>,
    high_compression: bool,
) -> crate::error::Result<()> {
    let attr_column_indices: Vec<usize> = key_list
        .iter()
        .map(|key| {
            graph
                .numeric_column_index(key)
                .unwrap_or_else(|| panic!("key {:?} not pre-loaded on graph", key))
        })
        .collect();

    // One writer per key, each owning its output path. No file (and no tallies directory) is
    // created here: the writer defers that to the first decoded assignment, so a run that fails
    // before producing data leaves nothing on disk.
    // The original input path drives the per-key output names.
    let in_name = source.path().to_string_lossy();
    let mut writers: Vec<DistrictMetricWriter> = key_list
        .iter()
        .map(|key| {
            let tally_dir = build_tally_output_dir(&in_name, output_dir);
            let output_path = build_tally_output_path(&in_name, key, max_samples, output_dir);
            DistrictMetricWriter::new(
                Box::new(move || {
                    create_dir_all(&tally_dir)?;
                    File::create(output_path)
                }),
                parquet_compression(high_compression),
                PARQUET_BATCH_ROWS,
            )
        })
        .collect();

    run_pipeline(
        source,
        AssignmentLengthCheck::MatchesGraph(graph.node_count),
        // The pipeline enforces that the district set is identical for every plan, so the schema
        // each writer fixes from its first row holds for the whole run.
        "tally",
        |assignment, _n_reps| {
            let (totals, n_districts, observed) =
                tally_keys(&graph, assignment, &attr_column_indices)?;
            Ok((observed, (totals, n_districts, observed)))
        },
        |step, n_reps, accepted, (totals, n_districts, observed)| {
            let n_districts = n_districts as usize;
            for (key_index, writer) in writers.iter_mut().enumerate() {
                let offset = key_index * n_districts;
                writer.push_row(
                    step,
                    n_reps,
                    accepted,
                    (observed, &totals[offset..offset + n_districts]),
                )?;
            }
            Ok(())
        },
        show_progress,
        max_samples,
    )?;

    log::info!("Writing final output...");
    for writer in writers {
        writer.finish()?;
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
            adjacency: None,
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
