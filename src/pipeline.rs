//! Frame-parallel decode pipeline shared by the batched, per-sample metric modes (`tally-keys`,
//! `cut-edges`, `polsby-popper`, `region-*`, and `unique-plans`).
//!
//! The key insight: `binary-ensemble`'s `BenDecoder::next` does two things per record — a cheap
//! byte-level frame pop AND an expensive RLE expansion into a `Vec<u16>` assignment. The two can be
//! separated via `BenDecoder::into_frames()` + `decode_ben_line` + `rle_to_vec`, all public API.
//!
//! `run_pipeline` pops frames serially on the caller thread (fast), then hands each batch of frames
//! to rayon for parallel RLE decode + metric compute in one fused pass. Results come back in
//! BEN-file order and are forwarded to `on_row` along with the running `sample_count` /
//! `accepted_count`. The per-accepted-frame modes (`changed-assignments`, `extract-unique-plans`)
//! instead drive `run_sequential_accepted_frames`, which keeps frames serial for their cross-frame
//! state.
//!
//! Both drivers enforce the same two first-frame contracts, so no mode re-implements them:
//! assignment length (via [`AssignmentLengthCheck`]) and, when a `district_set_label` is given,
//! a fixed district set across the whole ensemble.

use crate::district::{observed_assignment_districts, validate_district_set_unchanged};
use crate::error::{BenError, Result};
use ben::decode::{count_samples_from_file, decode_ben_line, BenDecoder, BenFrame};
use ben::utils::rle_to_vec;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::ParquetCompression;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, Cursor};
use std::path::Path;

/// Parquet compression to write with. Snappy is fast and plenty compact for tally outputs; Brotli
/// only pays off when storage is the bottleneck.
pub fn parquet_compression(high: bool) -> ParquetCompression {
    if high {
        ParquetCompression::Brotli(None)
    } else {
        ParquetCompression::Snappy
    }
}

/// Default batch size for streaming Parquet row-group writes. Matches Polars' current fallback
/// row-group size (`512 * 512`) so the streaming path stays close to the library's non-streaming
/// behavior.
pub const PARQUET_BATCH_ROWS: usize = 512 * 512;

const BATCH: usize = 256;

/// How every decoded assignment's length is validated. There is deliberately no opt-out: every
/// mode checks one of these two contracts.
#[derive(Clone, Copy)]
pub enum AssignmentLengthCheck {
    /// Graph-driven modes: every frame must have exactly the graph's node count. A mismatched
    /// frame fails before the metric runs (a too-long assignment would otherwise be silently
    /// truncated, a too-short one would panic deep in a metric).
    MatchesGraph(usize),
    /// Graph-free modes: the first frame's length becomes the expectation for the rest of the
    /// file, so a corrupt mixed-length ensemble errors instead of being processed as-is.
    UniformWithinFile,
}

/// Serial enforcement of [`AssignmentLengthCheck`], in BEN-file order.
struct LengthGuard {
    check: AssignmentLengthCheck,
    established: Option<usize>,
}

impl LengthGuard {
    fn new(check: AssignmentLengthCheck) -> Self {
        Self {
            check,
            established: None,
        }
    }

    fn check(&mut self, actual: usize) -> Result<()> {
        match self.check {
            AssignmentLengthCheck::MatchesGraph(expected) => {
                if actual != expected {
                    return Err(BenError::AssignmentLength { actual, expected });
                }
            }
            AssignmentLengthCheck::UniformWithinFile => match self.established {
                None => self.established = Some(actual),
                Some(expected) => {
                    if actual != expected {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "assignment length changed from {} to {} within the BEN file; \
                                 every plan in an ensemble must assign the same node set",
                                expected, actual
                            ),
                        )
                        .into());
                    }
                }
            },
        }
        Ok(())
    }
}

/// First-frame district-set contract: when a label is given, the first checked frame's district
/// set becomes the expectation and every later frame must match it exactly. This is the single
/// chokepoint that enforces "every plan in the ensemble uses the same district labels"; modes that
/// are label-invariant pass no label and opt out.
struct DistrictSetGuard<'a> {
    label: Option<&'a str>,
    expected: Option<u128>,
}

impl<'a> DistrictSetGuard<'a> {
    fn new(label: Option<&'a str>) -> Self {
        Self {
            label,
            expected: None,
        }
    }

    /// Whether checking is enabled — lets callers skip computing the observed set entirely for
    /// label-invariant modes.
    fn is_active(&self) -> bool {
        self.label.is_some()
    }

    fn check(&mut self, observed: u128) -> Result<()> {
        if let Some(label) = self.label {
            match self.expected {
                None => self.expected = Some(observed),
                Some(expected) => validate_district_set_unchanged(observed, expected, label)?,
            }
        }
        Ok(())
    }
}

pub struct AcceptedFrame {
    pub accepted_count: u64,
    pub assignment: Vec<u16>,
    pub n_reps: u16,
}

fn make_progress_bar(total_samples: usize) -> ProgressBar {
    let progress_bar = ProgressBar::new(total_samples as u64);
    progress_bar.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise} ETA {eta}]",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    progress_bar
}

fn decode_frame(frame: &BenFrame) -> io::Result<Vec<u16>> {
    decode_ben_line(
        Cursor::new(&frame.raw_data),
        frame.max_val_bits,
        frame.max_len_bits,
        frame.n_bytes,
    )
    .map(rle_to_vec)
    .map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to decode BEN frame: {e:?}"),
        )
    })
}

fn process_batch<Row, P>(
    process: &P,
    graph_node_count: Option<usize>,
    frames: &[BenFrame],
) -> Result<Vec<(u16, usize, u128, Row)>>
where
    Row: Send,
    P: Fn(&[u16], u16) -> Result<(u128, Row)> + Sync,
{
    frames
        .par_iter()
        .map(|frame| -> Result<(u16, usize, u128, Row)> {
            let assignment = decode_frame(frame)?;
            if let Some(expected) = graph_node_count {
                // The `MatchesGraph` half of the length contract must hold BEFORE the metric
                // closure runs: every graph-driven metric indexes `assignment[node_idx]` while
                // iterating a graph-length container, so a too-short assignment would otherwise
                // panic deep in a metric with an opaque out-of-bounds index. The serial
                // `LengthGuard` downstream re-asserts the same contract; this parallel pre-check
                // exists only to fail before `process` touches the assignment.
                if assignment.len() != expected {
                    return Err(BenError::AssignmentLength {
                        actual: assignment.len(),
                        expected,
                    });
                }
            }
            // `process` returns its own district label set (folded into the single pass it already
            // makes over the assignment/edges), so `run_pipeline` can enforce the set stays fixed
            // across the ensemble without a second pass. Label-invariant modes return `0`.
            let (observed, row) = process(&assignment, frame.count)?;
            Ok((frame.count, assignment.len(), observed, row))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect()
}

/// Walk the BEN file once (fast — just counts frames) and return the total sample count (sum of
/// `frame.count` across all frames). Useful for Vec preallocation before calling `run_pipeline`.
pub fn count_samples(in_file: &str) -> io::Result<usize> {
    count_samples_from_file(Path::new(in_file), "ben")
}

/// Walk the BEN file once and return the number of **frames** (accepted records), independent of
/// repetition counts. For Standard BEN this equals the sample count; for MkvChain BEN it is ≤ the
/// sample count. Needed by modes (like changed-assignments) whose progress and output-sizing are
/// per-accepted-step, not per-sample.
pub fn count_frames(in_file: &str) -> io::Result<usize> {
    let file = File::open(in_file)?;
    let frames = BenDecoder::new(file)?.into_frames();
    let mut frame_count = 0usize;
    for frame in frames {
        frame?;
        frame_count += 1;
    }
    Ok(frame_count)
}

/// Drive `on_frame` over every accepted frame in order, with the same first-frame contracts as
/// `run_pipeline`: each frame's length is validated per `length_check`, and when
/// `district_set_label` is `Some` each frame's district set must match the first frame's. Both
/// checks run BEFORE `on_frame`, so a mode's cross-frame state never sees a contract-violating
/// assignment.
pub fn run_sequential_accepted_frames<F>(
    in_file: &str,
    total_frames: usize,
    max_frames: Option<usize>,
    length_check: AssignmentLengthCheck,
    district_set_label: Option<&str>,
    show_progress: bool,
    mut on_frame: F,
) -> Result<u64>
where
    F: FnMut(AcceptedFrame) -> Result<()>,
{
    let frame_limit = max_frames.unwrap_or(total_frames);
    let progress_bar = show_progress.then(|| make_progress_bar(frame_limit));

    let file = File::open(in_file)?;
    let decoder = BenDecoder::new(file).map_err(io::Error::from)?;
    let mut accepted_count = 0u64;
    let mut length_guard = LengthGuard::new(length_check);
    let mut district_guard = DistrictSetGuard::new(district_set_label);

    for record_res in decoder.take(frame_limit) {
        let (assignment, n_reps) = record_res?;
        length_guard.check(assignment.len())?;
        if district_guard.is_active() {
            district_guard.check(observed_assignment_districts(&assignment)?.1)?;
        }
        accepted_count += 1;
        on_frame(AcceptedFrame {
            accepted_count,
            assignment,
            n_reps,
        })?;

        if let Some(progress_bar) = &progress_bar {
            progress_bar.inc(1);
        }
    }

    if let Some(progress_bar) = progress_bar {
        progress_bar.finish_and_clear();
    }

    Ok(accepted_count)
}

/// Run a per-sample `process` closure over every record in `in_file`, with frame extraction serial
/// on the caller thread and RLE-decode + `process` fused in parallel across a rayon pool.
///
/// `on_row` is called in BEN-file order with `(sample_count, n_reps, accepted_count, row)` —
/// `sample_count` advances by `n_reps` (MkvChain frames can carry >1), `accepted_count` by 1.
///
/// `process` takes `(&[u16], u16)` = `(assignment, n_reps)` and is invoked inside the rayon pool;
/// it must be `Sync` and produce `Send` rows. It returns `(district_set, row)`: the `u128` is the
/// observed district label set for this assignment (folded into the pass the metric already makes),
/// or `0` for label-invariant modes that opt out of the fixed-set check.
///
/// `length_check` is [`AssignmentLengthCheck::MatchesGraph`] (the graph's node count) for
/// graph-driven modes — every decoded assignment is asserted to match it before `process` runs, so
/// a BEN file that disagrees with the graph fails loudly instead of mistallying — or
/// [`AssignmentLengthCheck::UniformWithinFile`] for modes that don't tie assignments to a graph
/// (e.g. unique plans, which only hashes the raw assignment); there, the first frame fixes the
/// expected length and the check runs before each row reaches `on_row`.
///
/// `district_set_label` enables the fixed-district-set invariant: when `Some(label)`, the first
/// frame's district label set is captured and every later frame's `district_set` must match it
/// exactly, otherwise the run fails with an error naming `label`. This is the single chokepoint
/// that enforces "every plan in the ensemble uses the same district labels" for every graph-driven
/// mode at once. Pass `None` for label-invariant modes (unique plans), which must not be
/// constrained this way.
///
/// The total sample count needed to size the progress bar is computed here (a single extra pass
/// over the file) only when `show_progress` is set, so `--no-progress` runs never pay for it.
pub fn run_pipeline<Row, P, F>(
    in_file: &str,
    length_check: AssignmentLengthCheck,
    district_set_label: Option<&str>,
    process: P,
    mut on_row: F,
    show_progress: bool,
) -> Result<()>
where
    Row: Send,
    P: Fn(&[u16], u16) -> Result<(u128, Row)> + Sync,
    F: FnMut(u64, u32, u64, Row) -> Result<()>,
{
    let ben_file = File::open(in_file)?;

    let basename = Path::new(in_file)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    log::info!("Reading {:?}...", basename);

    let progress_bar = if show_progress {
        Some(make_progress_bar(count_samples(in_file)?))
    } else {
        None
    };

    let frames = BenDecoder::new(&ben_file)
        .map_err(io::Error::from)?
        .into_frames();
    let mut frame_batch: Vec<BenFrame> = Vec::with_capacity(BATCH);

    let mut sample_count: u64 = 1;
    let mut accepted_count: u64 = 1;
    // The parallel pre-check inside `process_batch` only applies to `MatchesGraph`; the serial
    // guards below enforce both contracts in BEN-file order before each row reaches `on_row`.
    let graph_node_count = match length_check {
        AssignmentLengthCheck::MatchesGraph(node_count) => Some(node_count),
        AssignmentLengthCheck::UniformWithinFile => None,
    };
    let mut length_guard = LengthGuard::new(length_check);
    let mut district_guard = DistrictSetGuard::new(district_set_label);

    for frame_res in frames {
        frame_batch.push(frame_res?);
        if frame_batch.len() < BATCH {
            continue;
        }
        for (n_reps, assignment_len, observed, row) in
            process_batch(&process, graph_node_count, &frame_batch)?
        {
            length_guard.check(assignment_len)?;
            district_guard.check(observed)?;
            on_row(sample_count, n_reps as u32, accepted_count, row)?;
            sample_count += n_reps as u64;
            accepted_count += 1;
            // Advance by n_reps so MkvChain repetitions tick the bar correctly.
            if let Some(progress_bar) = &progress_bar {
                progress_bar.inc(n_reps as u64);
            }
        }
        frame_batch.clear();
    }

    if !frame_batch.is_empty() {
        for (n_reps, assignment_len, observed, row) in
            process_batch(&process, graph_node_count, &frame_batch)?
        {
            length_guard.check(assignment_len)?;
            district_guard.check(observed)?;
            on_row(sample_count, n_reps as u32, accepted_count, row)?;
            sample_count += n_reps as u64;
            accepted_count += 1;
            if let Some(progress_bar) = &progress_bar {
                progress_bar.inc(n_reps as u64);
            }
        }
    }

    if let Some(progress_bar) = progress_bar {
        progress_bar.finish_and_clear();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        count_frames, count_samples, run_pipeline, run_sequential_accepted_frames,
        AssignmentLengthCheck,
    };
    use ben::encode::BenEncoder;
    use ben::BenVariant;
    use tempfile::NamedTempFile;

    fn write_ben_file(variant: BenVariant, assignments: &[Vec<u16>]) -> NamedTempFile {
        let file = NamedTempFile::new().unwrap();
        let writer = std::fs::File::create(file.path()).unwrap();
        let mut encoder = BenEncoder::new(writer, variant);
        for assignment in assignments {
            encoder.write_assignment(assignment.clone()).unwrap();
        }
        encoder.finish().unwrap();
        file
    }

    #[test]
    fn count_samples_and_frames_diverge_for_mkvchain_repetitions() {
        let ben_file = write_ben_file(
            BenVariant::MkvChain,
            &[vec![1, 1, 2, 2], vec![1, 1, 2, 2], vec![2, 2, 1, 1]],
        );

        assert_eq!(count_samples(ben_file.path().to_str().unwrap()).unwrap(), 3);
        assert_eq!(count_frames(ben_file.path().to_str().unwrap()).unwrap(), 2);
    }

    #[test]
    fn run_pipeline_reports_steps_repetitions_and_acceptance_order() {
        let ben_file = write_ben_file(
            BenVariant::MkvChain,
            &[vec![1, 1, 2, 2], vec![1, 1, 2, 2], vec![2, 2, 1, 1]],
        );

        let mut rows = Vec::new();
        run_pipeline(
            ben_file.path().to_str().unwrap(),
            AssignmentLengthCheck::MatchesGraph(4),
            None,
            |assignment, n_reps| Ok((0u128, (assignment[0], n_reps))),
            |step, n_reps, accepted, row| {
                rows.push((step, n_reps, accepted, row));
                Ok(())
            },
            false,
        )
        .unwrap();

        assert_eq!(rows, vec![(1, 2, 1, (1, 2)), (3, 1, 2, (2, 1)),]);
    }

    #[test]
    fn run_sequential_accepted_frames_reports_frame_order_and_respects_cap() {
        let ben_file = write_ben_file(
            BenVariant::MkvChain,
            &[vec![1, 1, 2, 2], vec![1, 1, 2, 2], vec![2, 2, 1, 1]],
        );

        let mut rows = Vec::new();
        let consumed = run_sequential_accepted_frames(
            ben_file.path().to_str().unwrap(),
            2,
            Some(1),
            AssignmentLengthCheck::UniformWithinFile,
            None,
            false,
            |frame| {
                rows.push((frame.accepted_count, frame.n_reps, frame.assignment));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(consumed, 1);
        assert_eq!(rows, vec![(1, 2, vec![1, 1, 2, 2])]);
    }

    /// Run the pipeline over a single length-4 assignment, declaring a graph of `expected_len`
    /// nodes. Used by the mismatch tests below; both directions are kept as separate cases to pin
    /// that the contract is exact equality, not "at least this long".
    fn run_pipeline_with_expected_len(expected_len: usize) -> crate::error::Result<()> {
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![0, 1, 2, 1]]);
        run_pipeline(
            ben_file.path().to_str().unwrap(),
            AssignmentLengthCheck::MatchesGraph(expected_len),
            None,
            |assignment, _n_reps| Ok((0u128, assignment.len())),
            |_step, _n_reps, _accepted, _row| Ok(()),
            false,
        )
    }

    #[test]
    fn run_pipeline_errors_when_assignment_longer_than_graph() {
        // A too-long assignment (more entries than the graph has nodes) is rejected up front rather
        // than silently truncated by a metric.
        let err = run_pipeline_with_expected_len(3).unwrap_err();
        assert_eq!(
            err.to_string(),
            "BEN assignment has 4 entries but graph has 3 nodes"
        );
    }

    #[test]
    fn run_pipeline_errors_when_assignment_shorter_than_graph() {
        let err = run_pipeline_with_expected_len(5).unwrap_err();
        assert_eq!(
            err.to_string(),
            "BEN assignment has 4 entries but graph has 5 nodes"
        );
    }

    #[test]
    fn run_pipeline_uniform_length_rejects_mixed_lengths() {
        // Graph-free modes establish the expected length from the first frame; a later frame of a
        // different length is a corrupt ensemble and must error in the driver, not be silently
        // forwarded to the mode.
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![1, 1, 2, 2], vec![1, 2, 2]]);
        let mut seen = 0usize;
        let err = run_pipeline(
            ben_file.path().to_str().unwrap(),
            AssignmentLengthCheck::UniformWithinFile,
            None,
            |_assignment, _n_reps| Ok((0u128, ())),
            |_step, _n_reps, _accepted, _row| {
                seen += 1;
                Ok(())
            },
            false,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "assignment length changed from 4 to 3 within the BEN file; \
             every plan in an ensemble must assign the same node set"
        );
        assert_eq!(seen, 1, "only the first (contract-establishing) row passes");
    }

    #[test]
    fn run_pipeline_preserves_step_and_accepted_accounting_across_batch_boundary() {
        // Frames are popped serially and decoded in batches of `BATCH` (256). With > 2*BATCH
        // Standard frames this drives the mid-loop flush at `frame_batch.len() == BATCH` AND the
        // trailing partial-batch flush, plus the cross-batch accumulation of `sample_count` /
        // `accepted_count` — the 256-frame seam, where an off-by-one in `step`/`accepted_count`
        // could hide on real (million-frame) ensembles.
        let n = 600usize;
        // Encode each frame's index into its assignment so we can also assert in-order delivery
        // across the seam, not just the running counters.
        let assignments: Vec<Vec<u16>> = (0..n).map(|i| vec![i as u16, i as u16]).collect();
        let ben_file = write_ben_file(BenVariant::Standard, &assignments);

        let mut rows = Vec::new();
        run_pipeline(
            ben_file.path().to_str().unwrap(),
            AssignmentLengthCheck::MatchesGraph(2),
            None,
            |assignment, _n_reps| Ok((0u128, assignment[0])),
            |step, n_reps, accepted, marker| {
                rows.push((step, n_reps, accepted, marker));
                Ok(())
            },
            false,
        )
        .unwrap();

        // Standard BEN never coalesces, so n_reps == 1 and both `step` and `accepted_count` run
        // 1..=n with no gaps or repeats, and the per-frame marker arrives strictly in input order.
        let expected: Vec<(u64, u32, u64, u16)> = (0..n)
            .map(|i| ((i + 1) as u64, 1u32, (i + 1) as u64, i as u16))
            .collect();
        assert_eq!(rows, expected);
    }

    #[test]
    fn run_pipeline_rejects_changed_district_set_when_labelled() {
        // `process` reports the district set it captured (here straight from the assignment). First
        // frame establishes {1,2}; the second drops district 2. With a `district_set_label` set,
        // the pipeline rejects the ensemble at the single chokepoint — this is what makes
        // label-agnostic modes (cut-edges, region) enforce a fixed district set without
        // their own validation code.
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![1, 1, 2, 2], vec![1, 1, 1, 1]]);
        let err = run_pipeline(
            ben_file.path().to_str().unwrap(),
            AssignmentLengthCheck::MatchesGraph(4),
            Some("cut-edges"),
            |assignment, _n_reps| {
                let mut observed = 0u128;
                for &d in assignment {
                    observed |= 1u128 << d;
                }
                Ok((observed, ()))
            },
            |_step, _n_reps, _accepted, _row| Ok(()),
            false,
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "districts [2] from the first assignment are missing from a later plan; \
             every plan in the ensemble must use the same district labels to stream cut-edges output with a fixed schema"
        );
    }

    #[test]
    fn run_pipeline_skips_district_set_check_when_unlabelled() {
        // Label-invariant modes (unique plans) pass `None` and a `0` district set: a changing set
        // must NOT error. Both frames below would violate the labelled invariant if it applied.
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![1, 1, 2, 2], vec![3, 3, 1, 1]]);
        let mut seen = 0usize;
        run_pipeline(
            ben_file.path().to_str().unwrap(),
            AssignmentLengthCheck::UniformWithinFile,
            None,
            |_assignment, _n_reps| Ok((0u128, ())),
            |_step, _n_reps, _accepted, _row| {
                seen += 1;
                Ok(())
            },
            false,
        )
        .unwrap();
        assert_eq!(seen, 2);
    }

    #[test]
    fn run_sequential_rejects_mixed_lengths_before_on_frame() {
        // The sequential driver enforces the uniform-length contract itself, so cross-frame modes
        // (changed-assignments) can never zip mismatched assignments.
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![1, 1, 2, 2], vec![1, 2, 2]]);
        let mut seen = 0usize;
        let err = run_sequential_accepted_frames(
            ben_file.path().to_str().unwrap(),
            2,
            None,
            AssignmentLengthCheck::UniformWithinFile,
            None,
            false,
            |_frame| {
                seen += 1;
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "assignment length changed from 4 to 3 within the BEN file; \
             every plan in an ensemble must assign the same node set"
        );
        assert_eq!(seen, 1, "the offending frame never reaches on_frame");
    }

    #[test]
    fn run_sequential_rejects_changed_district_set_when_labelled() {
        // Same chokepoint as the parallel driver: the first frame establishes {1,2}, the second
        // drops district 2, and the labelled sequential run fails before on_frame sees it.
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![1, 1, 2, 2], vec![1, 1, 1, 1]]);
        let mut seen = 0usize;
        let err = run_sequential_accepted_frames(
            ben_file.path().to_str().unwrap(),
            2,
            None,
            AssignmentLengthCheck::UniformWithinFile,
            Some("changed-assignments"),
            false,
            |_frame| {
                seen += 1;
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(
            err.to_string(),
            "districts [2] from the first assignment are missing from a later plan; \
             every plan in the ensemble must use the same district labels to stream changed-assignments output with a fixed schema"
        );
        assert_eq!(seen, 1);
    }

    #[test]
    fn run_sequential_skips_district_set_check_when_unlabelled() {
        // extract-unique-plans is label-invariant: district sets may change freely when no label
        // is given.
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![1, 1, 2, 2], vec![3, 3, 1, 1]]);
        let mut seen = 0usize;
        run_sequential_accepted_frames(
            ben_file.path().to_str().unwrap(),
            2,
            None,
            AssignmentLengthCheck::UniformWithinFile,
            None,
            false,
            |_frame| {
                seen += 1;
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(seen, 2);
    }
}
