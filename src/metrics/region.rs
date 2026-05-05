use crate::graph::Graph;
use crate::pipeline::{count_samples, parquet_compression, run_pipeline, PARQUET_BATCH_ROWS};
use polars::prelude::*;
use std::fs::File;

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

/// Count splits (regions spanning >1 district) or pieces (sum of district-set
/// sizes over all regions) for a single assignment against a single pre-loaded
/// region column.
///
/// Dense bitset keyed by interned region id × district id: one
/// `Vec<u64>` of length `n_regions * words_per_region`. `words_per_region`
/// is `ceil(n_districts / 64)` — for typical FL runs (< 64 districts) each
/// region occupies exactly one u64, so the whole bitset is `n_regions * 8`
/// bytes and sits in L1. Replaces the per-sample
/// `Vec<HashSet<u16>>` allocations from Phase 2.
fn region_metric_for_key(
    graph: &Graph,
    assignment: &[u16],
    region_col_idx: usize,
    metric: RegionMetric,
) -> u32 {
    let col = &graph.region_columns[region_col_idx];
    let n_regions = graph.region_id_counts[region_col_idx] as usize;
    if n_regions == 0 {
        return 0;
    }

    let max_d = assignment.iter().copied().max().unwrap_or(0) as usize;
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

fn region_batch_to_df(
    metric_col_name: &str,
    sample_nums: &mut Vec<u64>,
    n_reps_nums: &mut Vec<u32>,
    accepted_nums: &mut Vec<u32>,
    metric_keys: &mut Vec<String>,
    metric_values: &mut Vec<u32>,
) -> PolarsResult<DataFrame> {
    DataFrame::new_infer_height(vec![
        Series::new("step".into(), std::mem::take(sample_nums)).into(),
        Series::new("n_reps".into(), std::mem::take(n_reps_nums)).into(),
        Series::new("accepted_count".into(), std::mem::take(accepted_nums)).into(),
        Series::new("region_key".into(), std::mem::take(metric_keys)).into(),
        Series::new(metric_col_name.into(), std::mem::take(metric_values)).into(),
    ])
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
            *graph
                .region_index
                .get(k)
                .unwrap_or_else(|| panic!("region key {:?} not pre-loaded on graph", k))
        })
        .collect();

    let total = count_samples(in_file_name)?;
    let metric_col_name = region_metric_col_name(metric);
    let mut file = File::create(out_file_name).unwrap_or_else(|_| {
        panic!(
            "Failed to create output file {:?}. The file may already exist.",
            out_file_name
        )
    });
    let empty_df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), Vec::<u64>::new()).into(),
        Series::new("n_reps".into(), Vec::<u32>::new()).into(),
        Series::new("accepted_count".into(), Vec::<u32>::new()).into(),
        Series::new("region_key".into(), Vec::<String>::new()).into(),
        Series::new(metric_col_name.into(), Vec::<u32>::new()).into(),
    ])?;
    let mut writer = ParquetWriter::new(&mut file)
        .with_compression(parquet_compression(high_compression))
        .batched(empty_df.schema())?;

    let cap = PARQUET_BATCH_ROWS * key_list.len();
    let mut sample_nums: Vec<u64> = Vec::with_capacity(cap);
    let mut n_reps_nums: Vec<u32> = Vec::with_capacity(cap);
    let mut accepted_nums: Vec<u32> = Vec::with_capacity(cap);
    let mut metric_keys: Vec<String> = Vec::with_capacity(cap);
    let mut metric_values: Vec<u32> = Vec::with_capacity(cap);

    run_pipeline(
        in_file_name,
        total,
        |assignment, _n_reps| {
            key_list
                .iter()
                .zip(region_col_indices.iter())
                .map(|(key, &col_idx)| {
                    (
                        key.clone(),
                        region_metric_for_key(&graph, assignment, col_idx, metric),
                    )
                })
                .collect::<Vec<(String, u32)>>()
        },
        |step, n_reps, accepted, counts| {
            for (key, count_val) in counts {
                sample_nums.push(step);
                n_reps_nums.push(n_reps);
                accepted_nums.push(accepted);
                metric_keys.push(key);
                metric_values.push(count_val);
            }
            if sample_nums.len() >= cap {
                let df = region_batch_to_df(
                    metric_col_name,
                    &mut sample_nums,
                    &mut n_reps_nums,
                    &mut accepted_nums,
                    &mut metric_keys,
                    &mut metric_values,
                )
                .expect("Unable to build region-metric batch DataFrame");
                writer
                    .write_batch(&df)
                    .expect("Unable to write region-metric batch");
            }
        },
        show_progress,
    )?;

    eprintln!("Writing final output...");
    if !sample_nums.is_empty() {
        let df = region_batch_to_df(
            metric_col_name,
            &mut sample_nums,
            &mut n_reps_nums,
            &mut accepted_nums,
            &mut metric_keys,
            &mut metric_values,
        )?;
        writer.write_batch(&df)?;
    }
    writer.finish()?;

    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{region_metric_for_key, RegionMetric};
    use crate::graph::Graph;
    use std::collections::HashMap;

    fn graph_with_region_column(region_column: Vec<Option<u32>>, region_count: u32) -> Graph {
        Graph {
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

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits),
            1
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces),
            3
        );
    }

    #[test]
    fn region_metric_handles_district_ids_across_word_boundaries() {
        let graph = graph_with_region_column(vec![Some(0), Some(0), Some(1)], 2);
        let assignment = vec![0, 64, 64];

        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Splits),
            1
        );
        assert_eq!(
            region_metric_for_key(&graph, &assignment, 0, RegionMetric::Pieces),
            3
        );
    }

    #[test]
    fn region_metric_returns_zero_when_no_regions_are_present() {
        let graph = graph_with_region_column(vec![None, None], 0);
        assert_eq!(
            region_metric_for_key(&graph, &[1, 2], 0, RegionMetric::Splits),
            0
        );
        assert_eq!(
            region_metric_for_key(&graph, &[1, 2], 0, RegionMetric::Pieces),
            0
        );
    }
}
