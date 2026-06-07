use crate::district::observed_assignment_districts;
use crate::graph::Graph;
use crate::output::parquet::U32KeyedMetricWriter;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use std::fs::File;
use std::io;

#[derive(Clone, Copy)]
pub enum RegionMetric {
    Splits,
    Pieces,
}

fn region_metric_col_name(metric: RegionMetric) -> &'static str {
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
/// `n_regions * 8` bytes and sits in L1. Replaces the per-sample `Vec<HashSet<u16>>` allocations
/// from Phase 2.
/// `max_d` is the maximum district id in the assignment, computed once by the caller (together with
/// the observed-district set) so it isn't re-derived per region key.
fn region_metric_for_key(
    graph: &Graph,
    assignment: &[u16],
    region_col_idx: usize,
    metric: RegionMetric,
    max_d: usize,
) -> u32 {
    let col = &graph.region_columns[region_col_idx];
    let n_regions = graph.region_id_counts[region_col_idx] as usize;
    if n_regions == 0 {
        return 0;
    }

    let words_per_region = (max_d / 64) + 1;
    let mut bitset = vec![0u64; n_regions * words_per_region];

    for (i, maybe_rid) in col.iter().enumerate() {
        if let Some(rid) = *maybe_rid {
            let d = assignment[i] as usize;
            let w = rid as usize * words_per_region + (d >> 6);
            bitset[w] |= 1u64 << (d & 63);
        }
    }

    match metric {
        RegionMetric::Splits => (0..n_regions)
            .filter(|&r| {
                let start = r * words_per_region;
                let popcount: u32 = bitset[start..start + words_per_region]
                    .iter()
                    .map(|w| w.count_ones())
                    .sum();
                popcount > 1
            })
            .count() as u32,
        RegionMetric::Pieces => (0..n_regions)
            .map(|r| {
                let start = r * words_per_region;
                bitset[start..start + words_per_region]
                    .iter()
                    .map(|w| w.count_ones())
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
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let region_col_indices: Vec<usize> = key_list
        .iter()
        .map(|k| {
            graph
                .region_column_index(k)
                .unwrap_or_else(|| panic!("region key {:?} not pre-loaded on graph", k))
        })
        .collect();

    let total = count_samples(in_file_name)?;
    let metric_col_name = region_metric_col_name(metric);
    let file = File::create(out_file_name).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("failed to create region output file {out_file_name:?}: {e}"),
        )
    })?;

    let cap = PARQUET_BATCH_ROWS * key_list.len();
    let mut writer = U32KeyedMetricWriter::new(
        file,
        "region_key",
        metric_col_name,
        parquet_compression(high_compression),
        cap,
    )?;

    run_pipeline(
        in_file_name,
        total,
        Some(graph.node_count),
        // The pipeline enforces a fixed district set across the ensemble for region modes too.
        Some(metric_col_name),
        |assignment, _n_reps| {
            // One pass yields both the observed district set (for the pipeline's fixed-set check)
            // and `max_d`, which every per-key bitset below is sized from.
            let (n_districts, observed) = observed_assignment_districts(assignment);
            let max_d = n_districts.saturating_sub(1) as usize;
            let rows = key_list
                .iter()
                .zip(region_col_indices.iter())
                .map(|(key, &col_idx)| {
                    (
                        key.clone(),
                        region_metric_for_key(&graph, assignment, col_idx, metric, max_d),
                    )
                })
                .collect::<Vec<(String, u32)>>();
            Ok((observed, rows))
        },
        |step, n_reps, accepted, counts| {
            for (key, count_val) in counts {
                writer
                    .push(step, n_reps, accepted, key, count_val)
                    .map_err(|e| io::Error::other(e.to_string()))?;
            }
            Ok(())
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    writer.finish()?;

    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{region_metric_col_name, region_metric_for_key, RegionMetric};
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
        let max_d = 3;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_d),
            1
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_d),
            3
        );
    }

    #[test]
    fn region_metric_handles_district_ids_across_word_boundaries() {
        let graph = graph_with_region_column(vec![Some(0), Some(0), Some(1)], 2);
        let assignment = vec![0, 64, 64];
        let max_d = 64;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_d),
            1
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_d),
            3
        );
    }

    #[test]
    fn region_metric_handles_single_district_plan() {
        // max_d == 0 → words_per_region = 1 (the floor of 0/64 still buys us one word). Every node
        // maps to district 0, so each region has exactly one piece and zero splits regardless of
        // how many regions exist.
        let graph = graph_with_region_column(vec![Some(0), Some(1), Some(0), Some(1)], 2);
        let assignment = vec![0u16, 0, 0, 0];
        let max_d = 0;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_d),
            0
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_d),
            2
        );
    }

    #[test]
    fn region_metric_collapses_when_every_node_is_same_region_and_district() {
        // Single region, single district → zero splits, one piece.
        let graph = graph_with_region_column(vec![Some(0), Some(0), Some(0)], 1);
        let assignment = vec![5u16, 5, 5];
        let max_d = 5;

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits, max_d),
            0
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces, max_d),
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
        let metric_col_name = region_metric_col_name(RegionMetric::Splits);
        let mut writer = U32KeyedMetricWriter::new(
            File::create(file.path()).unwrap(),
            "region_key",
            metric_col_name,
            parquet_compression(false),
            2,
        )
        .unwrap();

        writer.push(1, 1, 1, "county", 2).unwrap();
        writer.push(2, 1, 2, "county", 3).unwrap();
        writer.push(3, 2, 3, "county", 4).unwrap();
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
            df.column(metric_col_name)
                .unwrap()
                .u32()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }
}
