use crate::district::{
    observe_district, observed_assignment_districts, validate_district_set_unchanged, MAX_DISTRICTS,
};
use crate::graph::Graph;
use crate::input::BenSource;
use crate::output::parquet::U32KeyedMetricWriter;
use crate::pipeline::{
    capped_reps, make_progress_bar, parquet_compression, run_pipeline, AssignmentLengthCheck,
    PARQUET_BATCH_ROWS,
};
use ben::io::reader::TwoDeltaFrameEvent;
use ben::BenVariant;
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

struct RegionKeyState {
    counts: Vec<u32>,
    distinct_counts: Vec<u16>,
    splits: u32,
    pieces: u32,
}

impl RegionKeyState {
    fn new(n_regions: usize) -> Self {
        Self {
            counts: vec![0; n_regions * MAX_DISTRICTS as usize],
            distinct_counts: vec![0; n_regions],
            splits: 0,
            pieces: 0,
        }
    }

    fn reset(&mut self) {
        self.counts.fill(0);
        self.distinct_counts.fill(0);
        self.splits = 0;
        self.pieces = 0;
    }

    fn add(&mut self, region: u32, district: u16) {
        let region = region as usize;
        let district = district as usize;
        let count = &mut self.counts[region * MAX_DISTRICTS as usize + district];
        if *count == 0 {
            if self.distinct_counts[region] == 1 {
                self.splits += 1;
            }
            self.distinct_counts[region] += 1;
            self.pieces += 1;
        }
        *count += 1;
    }

    fn remove(&mut self, region: u32, district: u16) {
        let region = region as usize;
        let district = district as usize;
        let count = &mut self.counts[region * MAX_DISTRICTS as usize + district];
        *count -= 1;
        if *count == 0 {
            if self.distinct_counts[region] == 2 {
                self.splits -= 1;
            }
            self.distinct_counts[region] -= 1;
            self.pieces -= 1;
        }
    }

    fn value(&self, metric: RegionMetric) -> u32 {
        match metric {
            RegionMetric::Splits => self.splits,
            RegionMetric::Pieces => self.pieces,
        }
    }
}

/// Maintains region split/piece counts across TwoDelta events.
///
/// `update_delta` expects `before` to still be the pre-delta assignment; the caller applies the
/// changes after region counts and district counts are patched.
struct IncrementalRegionMetrics<'g> {
    graph: &'g Graph,
    region_column_indices: &'g [usize],
    key_states: Vec<RegionKeyState>,
    node_counts: Vec<u32>,
    observed: u128,
}

impl<'g> IncrementalRegionMetrics<'g> {
    fn new(graph: &'g Graph, region_column_indices: &'g [usize]) -> Self {
        let key_states = region_column_indices
            .iter()
            .map(|&column_index| RegionKeyState::new(graph.region_id_counts[column_index] as usize))
            .collect();
        Self {
            graph,
            region_column_indices,
            key_states,
            node_counts: vec![0; MAX_DISTRICTS as usize],
            observed: 0,
        }
    }

    /// Recompute all region counts and district counts from a snapshot assignment.
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        self.node_counts.fill(0);
        self.observed = 0;
        for key_state in &mut self.key_states {
            key_state.reset();
        }

        for (node, &district) in assignment.iter().enumerate() {
            observe_district(&mut self.observed, district)?;
            self.node_counts[district as usize] += 1;
            for (&column_index, key_state) in self
                .region_column_indices
                .iter()
                .zip(self.key_states.iter_mut())
            {
                if let Some(region) = self.graph.region_columns[column_index][node] {
                    key_state.add(region, district);
                }
            }
        }

        Ok(())
    }

    /// Apply one delta event to the maintained region metrics and district set.
    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[(usize, u16, u16)],
    ) -> crate::error::Result<()> {
        for &(node, old, new) in changes {
            let Some(&current) = before.get(node) else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "TwoDelta delta references node {node} outside assignment length {}",
                        before.len()
                    ),
                )
                .into());
            };
            if current != old {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "TwoDelta delta old label mismatch at node {node}: \
                         expected {current}, got {old}",
                    ),
                )
                .into());
            }

            observe_district(&mut self.observed, old)?;
            observe_district(&mut self.observed, new)?;
            if old == new {
                continue;
            }

            self.node_counts[new as usize] += 1;
            self.node_counts[old as usize] -= 1;
            if self.node_counts[old as usize] == 0 {
                self.observed &= !(1u128 << old);
            }

            for (&column_index, key_state) in self
                .region_column_indices
                .iter()
                .zip(self.key_states.iter_mut())
            {
                if let Some(region) = self.graph.region_columns[column_index][node] {
                    key_state.remove(region, old);
                    key_state.add(region, new);
                }
            }
        }

        Ok(())
    }
}

fn push_region_rows(
    writer: &mut U32KeyedMetricWriter,
    key_list: &[String],
    step: u64,
    n_reps: u32,
    accepted: u64,
    values: impl Iterator<Item = u32>,
) -> crate::error::Result<()> {
    for (key, value) in key_list.iter().zip(values) {
        writer.push_row(step, n_reps, accepted, (key.clone(), value))?;
    }
    Ok(())
}

/// Run region metrics directly from TwoDelta events, reseeding on snapshots and patching deltas.
#[allow(clippy::too_many_arguments)]
fn run_incremental_twodelta_region_metric(
    graph: &Graph,
    source: &BenSource,
    writer: &mut U32KeyedMetricWriter,
    key_list: &[String],
    region_column_indices: &[usize],
    metric: RegionMetric,
    metric_column_name: &str,
    show_progress: bool,
    max_samples: Option<usize>,
) -> crate::error::Result<()> {
    let progress_bar = if show_progress {
        Some(make_progress_bar(match max_samples {
            Some(n) => n,
            None => source.count_samples()?,
        }))
    } else {
        None
    };

    let mut remaining_samples = max_samples;
    let mut assignment: Option<Vec<u16>> = None;
    let mut expected_observed: Option<u128> = None;
    let mut state = IncrementalRegionMetrics::new(graph, region_column_indices);
    let mut step = 1u64;

    for (accepted, event) in (1u64..).zip(source.open_reader()?.into_twodelta_events()) {
        if remaining_samples == Some(0) {
            break;
        }

        let n_reps = match event? {
            TwoDeltaFrameEvent::Snapshot {
                assignment: snapshot,
                count,
                ..
            } => {
                if snapshot.len() != graph.node_count {
                    return Err(crate::error::Error::AssignmentLength {
                        actual: snapshot.len(),
                        expected: graph.node_count,
                    });
                }
                state.seed(&snapshot)?;
                assignment = Some(snapshot);
                capped_reps(&mut remaining_samples, count)
            }
            TwoDeltaFrameEvent::Delta { changes, count } => {
                let assignment = assignment.as_mut().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "TwoDelta delta event appeared before an initial snapshot",
                    )
                })?;
                let changes = changes
                    .into_iter()
                    .map(|(node, old, new)| (node as usize, old, new))
                    .collect::<Vec<_>>();
                state.update_delta(assignment, &changes)?;
                for (node, _old, new) in changes {
                    assignment[node] = new;
                }
                capped_reps(&mut remaining_samples, count)
            }
        };

        match expected_observed {
            None => expected_observed = Some(state.observed),
            Some(expected) => {
                validate_district_set_unchanged(state.observed, expected, metric_column_name)?;
            }
        }

        push_region_rows(
            writer,
            key_list,
            step,
            n_reps as u32,
            accepted,
            state
                .key_states
                .iter()
                .map(|key_state| key_state.value(metric)),
        )?;
        step += n_reps as u64;
        if let Some(progress_bar) = &progress_bar {
            progress_bar.inc(n_reps as u64);
        }
    }

    if let Some(progress_bar) = progress_bar {
        progress_bar.finish_and_clear();
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn tally_and_save_region_metric(
    graph: Graph,
    source: &BenSource,
    out_file_name: &str,
    key_list: Vec<String>,
    metric: RegionMetric,
    show_progress: bool,
    max_samples: Option<usize>,
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

    if source.variant()? == BenVariant::TwoDelta {
        run_incremental_twodelta_region_metric(
            &graph,
            source,
            &mut writer,
            &key_list,
            &region_column_indices,
            metric,
            metric_column_name,
            show_progress,
            max_samples,
        )?;
    } else {
        run_pipeline(
            source,
            AssignmentLengthCheck::MatchesGraph(graph.node_count),
            // The pipeline enforces a fixed district set across the ensemble for region modes too.
            metric_column_name,
            |assignment, _n_reps| {
                // One pass yields both the observed district set (for the pipeline's fixed-set
                // check) and `max_district`, which every per-key bitset below is sized from.
                let (n_districts, observed) = observed_assignment_districts(assignment)?;
                let max_district = n_districts.saturating_sub(1) as usize;
                let rows = region_column_indices
                    .iter()
                    .map(|&column_index| {
                        region_metric_for_key(
                            &graph,
                            assignment,
                            column_index,
                            metric,
                            max_district,
                        )
                    })
                    .collect::<Vec<u32>>();
                Ok((observed, rows))
            },
            |step, n_reps, accepted, counts| {
                push_region_rows(
                    &mut writer,
                    &key_list,
                    step,
                    n_reps,
                    accepted,
                    counts.into_iter(),
                )
            },
            show_progress,
            max_samples,
        )?;
    }

    log::info!("Writing final output...");
    writer.finish()?;

    log::info!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        region_metric_column_name, region_metric_for_key, IncrementalRegionMetrics, RegionMetric,
    };
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
            adjacency: None,
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
    fn incremental_region_metrics_rejects_delta_old_label_mismatch() {
        let graph = graph_with_region_column(vec![Some(0), Some(0), Some(1)], 2);
        let before = vec![1, 1, 2];
        let changes = vec![(1usize, 2u16, 1u16)];
        let mut state = IncrementalRegionMetrics::new(&graph, &[0]);

        state.seed(&before).unwrap();
        let err = state.update_delta(&before, &changes).unwrap_err();

        assert!(
            err.to_string().contains("old label mismatch"),
            "unexpected error: {err}",
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
