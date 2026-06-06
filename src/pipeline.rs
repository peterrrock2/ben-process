//! Frame-parallel decode pipeline shared by the three batched metric modes
//! (`tally-keys`, `cut-edges`, `region-*`).
//!
//! The key insight: `binary-ensemble`'s `BenDecoder::next` does two things per record — a cheap
//! byte-level frame pop AND an expensive RLE expansion into a `Vec<u16>` assignment. The two can be
//! separated via `BenDecoder::into_frames()` (returns a `BenFrameDecoeder`) + `decode_ben_line` +
//! `rle_to_vec`, both of which are public API.
//!
//! `run_pipeline` pops frames serially on the caller thread (fast), then hands each batch of frames
//! to rayon for parallel RLE decode + metric compute in one fused pass. Results come back in
//! BEN-file order and are forwarded to `on_row` along with the running `sample_count` /
//! `accepted_count` (same semantics as the pre-Phase-4 loops).

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
    let pb = ProgressBar::new(total_samples as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise} ETA {eta}]",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb
}

fn decode_frame(frame: &BenFrame) -> Vec<u16> {
    decode_ben_line(
        Cursor::new(&frame.raw_data),
        frame.max_val_bits,
        frame.max_len_bits,
        frame.n_bytes,
    )
    .map(rle_to_vec)
    .expect("Failed to decode BEN frame")
}

fn process_batch<Row, P>(
    process: &P,
    expected_assignment_len: Option<usize>,
    frames: &[BenFrame],
) -> Vec<(u16, Row)>
where
    Row: Send,
    P: Fn(&[u16], u16) -> Row + Sync,
{
    frames
        .par_iter()
        .map(|frame| {
            let assignment = decode_frame(frame);
            if let Some(expected) = expected_assignment_len {
                // Every graph-driven metric indexes `assignment[node_idx]` while iterating a
                // graph-length container. A too-long assignment would otherwise be silently
                // truncated to the first `expected` entries (wrong-but-quiet tallies); a too-short
                // one would panic deep in a metric with an opaque out-of-bounds index. Checking
                // here, at the single point every assignment is decoded, fixes both directions for
                // all modes at once.
                assert_eq!(
                    assignment.len(),
                    expected,
                    "BEN assignment has {} entries but graph has {} nodes",
                    assignment.len(),
                    expected,
                );
            }
            let row = process(&assignment, frame.count);
            (frame.count, row)
        })
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
    let mut n = 0usize;
    for frame in frames {
        frame?;
        n += 1;
    }
    Ok(n)
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
    let pb = show_progress.then(|| make_progress_bar(frame_limit));

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

        if let Some(pb) = &pb {
            pb.inc(1);
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
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
/// it must be `Sync` and produce `Send` rows.
///
/// `expected_assignment_len` is the graph's node count for graph-driven modes; every decoded
/// assignment is asserted to match it before `process` runs, so a BEN file that disagrees with the
/// graph fails loudly instead of mistallying. Pass `None` for modes that don't tie assignments to a
/// graph (e.g. unique plans, which only hashes the raw assignment).
pub fn run_pipeline<Row, P, F>(
    in_file: &str,
    total_samples: usize,
    expected_assignment_len: Option<usize>,
    process: P,
    mut on_row: F,
    show_progress: bool,
) -> io::Result<()>
where
    Row: Send,
    P: Fn(&[u16], u16) -> Row + Sync,
    F: FnMut(u64, u32, u32, Row),
{
    let ben_file = File::open(in_file)?;

    let basename = Path::new(in_file)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    eprintln!("Reading {:?}...", basename);

    let pb = show_progress.then(|| make_progress_bar(total_samples));

    let frames = BenDecoder::new(&ben_file)?.into_frames();
    let mut frame_batch: Vec<BenFrame> = Vec::with_capacity(BATCH);

    let mut sample_count: u64 = 1;
    let mut accepted_count: u32 = 1;

    for frame_res in frames {
        frame_batch.push(frame_res?);
        if frame_batch.len() < BATCH {
            continue;
        }
        for (n_reps, row) in process_batch(&process, expected_assignment_len, &frame_batch) {
            on_row(sample_count, n_reps as u32, accepted_count, row);
            sample_count += n_reps as u64;
            accepted_count += 1;
            // Advance by n_reps so MkvChain repetitions tick the bar correctly.
            if let Some(pb) = &pb {
                pb.inc(n_reps as u64);
            }
        }
        frame_batch.clear();
    }

    if !frame_batch.is_empty() {
        for (n_reps, row) in process_batch(&process, expected_assignment_len, &frame_batch) {
            on_row(sample_count, n_reps as u32, accepted_count, row);
            sample_count += n_reps as u64;
            accepted_count += 1;
            if let Some(pb) = &pb {
                pb.inc(n_reps as u64);
            }
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{count_frames, count_samples, run_pipeline, run_sequential_accepted_frames};
    use ben::encode::BenEncoder;
    use ben::BenVariant;
    use std::error::Error;
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
            3,
            Some(4),
            |assignment, n_reps| (assignment[0], n_reps),
            |step, n_reps, accepted, row| rows.push((step, n_reps, accepted, row)),
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
    fn run_pipeline_with_expected_len(expected_len: usize) {
        let ben_file = write_ben_file(BenVariant::Standard, &[vec![0, 1, 2, 1]]);
        run_pipeline(
            ben_file.path().to_str().unwrap(),
            1,
            Some(expected_len),
            |assignment, _n_reps| assignment.len(),
            |_step, _n_reps, _accepted, _row| {},
            false,
        )
        .unwrap();
    }

    #[test]
    #[should_panic(expected = "BEN assignment has 4 entries but graph has 3 nodes")]
    fn run_pipeline_panics_when_assignment_longer_than_graph() {
        // A BEN file whose assignments are longer than the graph used to be silently truncated by
        // every graph-driven metric. The pipeline now rejects the mismatch up front for all modes
        // at once.
        run_pipeline_with_expected_len(3);
    }

    #[test]
    #[should_panic(expected = "BEN assignment has 4 entries but graph has 5 nodes")]
    fn run_pipeline_panics_when_assignment_shorter_than_graph() {
        run_pipeline_with_expected_len(5);
    }
}
