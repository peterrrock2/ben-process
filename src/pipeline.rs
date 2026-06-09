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

use crate::district::validate_district_set_unchanged;
use ben::decode::{count_samples_from_file, decode_ben_line, BenDecoder, BenFrame};
use ben::utils::rle_to_vec;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::ParquetCompression;
use rayon::prelude::*;
use std::error::Error;
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

pub struct AcceptedFrame {
    pub accepted_count: u32,
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
    expected_assignment_len: Option<usize>,
    frames: &[BenFrame],
) -> io::Result<Vec<(u16, u128, Row)>>
where
    Row: Send,
    P: Fn(&[u16], u16) -> io::Result<(u128, Row)> + Sync,
{
    frames
        .par_iter()
        .map(|frame| -> io::Result<(u16, u128, Row)> {
            let assignment = decode_frame(frame)?;
            if let Some(expected) = expected_assignment_len {
                // Every graph-driven metric indexes `assignment[node_idx]` while iterating a
                // graph-length container. A too-long assignment would otherwise be silently
                // truncated to the first `expected` entries (wrong-but-quiet tallies); a too-short
                // one would panic deep in a metric with an opaque out-of-bounds index. Checking
                // here, at the single point every assignment is decoded, fixes both directions for
                // all modes at once.
                if assignment.len() != expected {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "BEN assignment has {} entries but graph has {} nodes",
                            assignment.len(),
                            expected
                        ),
                    ));
                }
            }
            // `process` returns its own district label set (folded into the single pass it already
            // makes over the assignment/edges), so `run_pipeline` can enforce the set stays fixed
            // across the ensemble without a second pass. Label-invariant modes return `0`.
            let (observed, row) = process(&assignment, frame.count)?;
            Ok((frame.count, observed, row))
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

pub fn run_sequential_accepted_frames<F>(
    in_file: &str,
    total_frames: usize,
    max_frames: Option<usize>,
    show_progress: bool,
    mut on_frame: F,
) -> Result<u64, Box<dyn Error>>
where
    F: FnMut(AcceptedFrame) -> Result<(), Box<dyn Error>>,
{
    let frame_limit = max_frames.unwrap_or(total_frames);
    let progress_bar = show_progress.then(|| make_progress_bar(frame_limit));

    let file = File::open(in_file)?;
    let decoder = BenDecoder::new(file)?;
    let mut accepted_count = 0u64;

    for record_res in decoder.take(frame_limit) {
        let (assignment, n_reps) = record_res?;
        accepted_count += 1;
        on_frame(AcceptedFrame {
            accepted_count: accepted_count as u32,
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
/// `expected_assignment_len` is the graph's node count for graph-driven modes; every decoded
/// assignment is asserted to match it before `process` runs, so a BEN file that disagrees with the
/// graph fails loudly instead of mistallying. Pass `None` for modes that don't tie assignments to a
/// graph (e.g. unique plans, which only hashes the raw assignment).
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
    expected_assignment_len: Option<usize>,
    district_set_label: Option<&str>,
    process: P,
    mut on_row: F,
    show_progress: bool,
) -> io::Result<()>
where
    Row: Send,
    P: Fn(&[u16], u16) -> io::Result<(u128, Row)> + Sync,
    F: FnMut(u64, u32, u32, Row) -> io::Result<()>,
{
    let ben_file = File::open(in_file)?;

    let basename = Path::new(in_file)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    eprintln!("Reading {:?}...", basename);

    let progress_bar = if show_progress {
        Some(make_progress_bar(count_samples(in_file)?))
    } else {
        None
    };

    let frames = BenDecoder::new(&ben_file)?.into_frames();
    let mut frame_batch: Vec<BenFrame> = Vec::with_capacity(BATCH);

    let mut sample_count: u64 = 1;
    let mut accepted_count: u32 = 1;
    // The first frame's district label set; every later frame must match it when enforcing.
    let mut expected_district_set: Option<u128> = None;

    // Validate one frame's district set against the established expectation (establishing it on the
    // first frame), in BEN-file order, before its row is handed to `on_row`.
    let mut check_district_set = |observed: u128| -> io::Result<()> {
        if let Some(label) = district_set_label {
            match expected_district_set {
                None => expected_district_set = Some(observed),
                Some(expected) => validate_district_set_unchanged(observed, expected, label)?,
            }
        }
        Ok(())
    };

    for frame_res in frames {
        frame_batch.push(frame_res?);
        if frame_batch.len() < BATCH {
            continue;
        }
        for (n_reps, observed, row) in
            process_batch(&process, expected_assignment_len, &frame_batch)?
        {
            check_district_set(observed)?;
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
        for (n_reps, observed, row) in
            process_batch(&process, expected_assignment_len, &frame_batch)?
        {
            check_district_set(observed)?;
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
    use super::{count_frames, count_samples, run_pipeline, run_sequential_accepted_frames};
    use ben::encode::BenEncoder;
    use ben::BenVariant;
    use std::error::Error;
    use std::io;
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
            Some(4),
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
            false,
            |frame| {
                rows.push((frame.accepted_count, frame.n_reps, frame.assignment));
                Ok::<(), Box<dyn Error>>(())
            },
        )
        .unwrap();

        assert_eq!(consumed, 1);
        assert_eq!(rows, vec![(1, 2, vec![1, 1, 2, 2])]);
    }

    /// Run the pipeline over a single length-4 assignment, declaring a graph of `expected_len`
    /// nodes. Used by the mismatch tests below; both directions are kept as separate cases to pin
    /// that the contract is exact equality, not "at least this long".
    fn run_pipeline_with_expected_len(expected_len: usize) -> io::Result<()> {
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![0, 1, 2, 1]]);
        run_pipeline(
            ben_file.path().to_str().unwrap(),
            Some(expected_len),
            None,
            |assignment, _n_reps| Ok((0u128, assignment.len())),
            |_step, _n_reps, _accepted, _row| Ok(()),
            false,
        )
    }

    #[test]
    fn run_pipeline_errors_when_assignment_longer_than_graph() {
        // A BEN file whose assignments are longer than the graph used to be silently truncated by
        // every graph-driven metric. The pipeline now rejects the mismatch up front for all modes
        // at once.
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
    fn run_pipeline_preserves_step_and_accepted_accounting_across_batch_boundary() {
        // Frames are popped serially and decoded in batches of `BATCH` (256). With > 2*BATCH
        // Standard frames this drives the mid-loop flush at `frame_batch.len() == BATCH` AND the
        // trailing partial-batch flush, plus the cross-batch accumulation of `sample_count` /
        // `accepted_count`. Every other test uses < BATCH frames, so the 256-seam accounting was
        // previously unexercised — exactly where an off-by-one in `step`/`accepted_count` would
        // hide on real (million-frame) ensembles while the rest of the suite stayed green.
        let n = 600usize;
        // Encode each frame's index into its assignment so we can also assert in-order delivery
        // across the seam, not just the running counters.
        let assignments: Vec<Vec<u16>> = (0..n).map(|i| vec![i as u16, i as u16]).collect();
        let ben_file = write_ben_file(BenVariant::Standard, &assignments);

        let mut rows = Vec::new();
        run_pipeline(
            ben_file.path().to_str().unwrap(),
            Some(2),
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
        let expected: Vec<(u64, u32, u32, u16)> = (0..n)
            .map(|i| ((i + 1) as u64, 1u32, (i + 1) as u32, i as u16))
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
            Some(4),
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
            None,
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
}
