use crate::district::observed_assignment_districts;
use crate::graph::Graph;
use crate::output::parquet::U32KeyedMetricWriter;
use crate::pipeline::{parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use std::fs::File;
use std::io;

#[derive(Clone, Copy)]
pub enum RegionMetric {
    Splits,
    Pieces,
}

fn region_metric_column_name(metric: RegionMetric) -> &'static str {
    match metric {
        RegionMetric::Splits => "region_splits",
        RegionMetric::Pieces => "region_pieces",
    }
}

/// Count splits (regions spanning >1 district) or pieces (sum of district-set sizes over all
/// regions) for a single assignment against a single pre-loaded region column.
///
/// Dense bitset keyed by interned region id × district id: one `Vec<u64>` of length
/// `n_regions * words_per_region`. `words_per_region` is `ceil(n_districts / 64)` — for typical FL
/// runs (< 64 districts) each region occupies exactly one u64, so the whole bitset is
/// `n_regions * 8` bytes and sits in L1.
///
/// `max_district` is the maximum district id in the assignment, computed once by the caller
/// (together with the observed-district set) so it isn't re-derived per region key.
fn region_metric_for_key(
    graph: &Graph,
    assignment: &[u16],
    region_column_index: usize,
    metric: RegionMetric,
    max_district: usize,
) -> u32 {
    let column = &graph.region_columns[region_column_index];
    let n_regions = graph.region_id_counts[region_column_index] as usize;
    if n_regions == 0 {
        return 0;
    }

    let words_per_region = (max_district / 64) + 1;
    let mut bitset = vec![0u64; n_regions * words_per_region];

    for (node_index, maybe_region_id) in column.iter().enumerate() {
        if let Some(region_id) = *maybe_region_id {
            let district = assignment[node_index] as usize;
            let word_index = region_id as usize * words_per_region + (district >> 6);
            bitset[word_index] |= 1u64 << (district & 63);
        }
    }

    match metric {
        RegionMetric::Splits => (0..n_regions)
            .filter(|&region_id| {
                let region_start = region_id * words_per_region;
                let popcount: u32 = bitset[region_start..region_start + words_per_region]
                    .iter()
                    .map(|word| word.count_ones())
                    .sum();
                popcount > 1
            })
            .count() as u32,
        RegionMetric::Pieces => (0..n_regions)
            .map(|region_id| {
                let region_start = region_id * words_per_region;
                bitset[region_start..region_start + words_per_region]
                    .iter()
                    .map(|word| word.count_ones())
                    .sum::<u32>()
            })
            .sum(),
    }
}

pub fn tally_and_save_region_metric(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    key_list: Vec<String>,
    metric: RegionMetric,
    show_progress: bool,
    high_compression: bool,
) -> crate::error::Result<()> {
    let region_column_indices: Vec<usize> = key_list
        .iter()
        .map(|key| {
            graph
                .region_column_index(key)
                .unwrap_or_else(|| panic!("region key {:?} not pre-loaded on graph", key))
        })
        .collect();

    let metric_column_name = region_metric_column_name(metric);
    // The output file is created lazily on the first decoded assignment (or at finish for a
    // zero-frame run), so a run that fails before producing data leaves nothing on disk.
    let out_path = out_file_name.to_string();
    let batch_capacity = PARQUET_BATCH_ROWS * key_list.len();
    let mut writer = U32KeyedMetricWriter::new(
        Box::new(move || {
            File::create(&out_path).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("failed to create region output file {out_path:?}: {e}"),
                )
            })
        }),
        "region_key",
        metric_column_name,
        parquet_compression(high_compression),
        batch_capacity,
    );

    run_pipeline(
        in_file_name,
        Some(graph.node_count),
        // The pipeline enforces a fixed district set across the ensemble for region modes too.
        Some(metric_column_name),
        |assignment, _n_reps| {
            // One pass yields both the observed district set (for the pipeline's fixed-set check)
            // and `max_district`, which every per-key bitset below is sized from.
            let (n_districts, observed) = observed_assignment_districts(assignment)?;
            let max_district = n_districts.saturating_sub(1) as usize;
            let rows = key_list
                .iter()
                .zip(region_column_indices.iter())
                .map(|(key, &column_index)| {
                    (
                        key.clone(),
                        region_metric_for_key(
                            &graph,
                            assignment,
                            column_index,
                            metric,
                            max_district,
                        ),
                    )
                })
                .collect::<Vec<(String, u32)>>();
            Ok((observed, rows))
        },
        |step, n_reps, accepted, counts| {
            for (key, count) in counts {
                writer.push_row(step, n_reps, accepted, (key, count))?;
            }
            Ok(())
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
    use super::{region_metric_column_name, region_metric_for_key, RegionMetric};
    use crate::graph::Graph;
    use crate::output::parquet::U32KeyedMetricWriter;
    use crate::pipeline::parquet_compression;
    use polars::prelude::{ParquetReader, SerReader};
    use std::collections::HashMap;
    use std::fs::File;
    use tempfile::NamedTempFile;

    fn graph_with_region_column(region_column: Vec<Option<u32>>, region_count: u32) -> Graph {
        Graph {
            node_count: region_column.len(),
            attr_columns: vec![],
            attr_index: HashMap::new(),
            region_columns: vec![region_column],
            region_index: HashMap::new(),
            region_id_counts: vec![region_count],
            edges: vec![],
            edge_weights: None,
        }
    }

    #[test]
    fn region_metric_counts_splits_and_pieces_while_ignoring_missing_regions() {
        let graph = graph_with_region_column(vec![Some(0), Some(0), Some(1), None], 2);
        let assignment = vec![1, 2, 2, 3];
        let max_district = 3;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_district),
            1
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_district),
            3
        );
    }

    #[test]
    fn region_metric_handles_district_ids_across_word_boundaries() {
        let graph = graph_with_region_column(vec![Some(0), Some(0), Some(1)], 2);
        let assignment = vec![0, 64, 64];
        let max_district = 64;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_district),
            1
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_district),
            3
        );
    }

    #[test]
    fn region_metric_handles_single_district_plan() {
        // max_district == 0 → words_per_region = 1 (the floor of 0/64 still buys us one word).
        // Every node maps to district 0, so each region has exactly one piece and zero
        // splits regardless of how many regions exist.
        let graph = graph_with_region_column(vec![Some(0), Some(1), Some(0), Some(1)], 2);
        let assignment = vec![0u16, 0, 0, 0];
        let max_district = 0;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_district),
            0
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_district),
            2
        );
    }

    #[test]
    fn region_metric_collapses_when_every_node_is_same_region_and_district() {
        // Single region, single district → zero splits, one piece.
        let graph = graph_with_region_column(vec![Some(0), Some(0), Some(0)], 1);
        let assignment = vec![5u16, 5, 5];
        let max_district = 5;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_district),
            0
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_district),
            1
        );
    }

    #[test]
    fn region_metric_returns_zero_when_no_regions_are_present() {
        let graph = graph_with_region_column(vec![None, None], 0);
        assert_eq!(
            region_metric_for_key(&graph, &[1, 2], 0, RegionMetric::Splits, 2),
            0
        );
        assert_eq!(
            region_metric_for_key(&graph, &[1, 2], 0, RegionMetric::Pieces, 2),
            0
        );
    }

    #[test]
    fn region_batched_writer_appends_multiple_batches() {
        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();
        let metric_column_name = region_metric_column_name(RegionMetric::Splits);
        let mut writer = U32KeyedMetricWriter::new(
            Box::new(move || File::create(path)),
            "region_key",
            metric_column_name,
            parquet_compression(false),
            2,
        );

        writer.push_row(1, 1, 1, ("county".into(), 2)).unwrap();
        writer.push_row(2, 1, 2, ("county".into(), 3)).unwrap();
        writer.push_row(3, 2, 3, ("county".into(), 4)).unwrap();
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
            df.column(metric_column_name)
                .unwrap()
                .u32()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }
}
