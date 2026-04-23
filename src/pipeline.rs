//! Frame-parallel decode pipeline shared by the three batched metric modes
//! (`tally-keys`, `cut-edges`, `region-*`).
//!
//! The key insight: `binary-ensemble`'s `BenDecoder::next` does two things
//! per record — a cheap byte-level frame pop AND an expensive RLE expansion
//! into a `Vec<u16>` assignment. The two can be separated via
//! `BenDecoder::into_frames()` (returns a `BenFrameDecoeder`) +
//! `decode_ben_line` + `rle_to_vec`, both of which are public API.
//!
//! `run_pipeline` pops frames serially on the caller thread (fast), then
//! hands each batch of frames to rayon for parallel RLE decode + metric
//! compute in one fused pass. Results come back in BEN-file order and are
//! forwarded to `on_row` along with the running `sample_count` /
//! `accepted_count` (same semantics as the pre-Phase-4 loops).

use ben::decode::{count_samples_from_file, decode_ben_line, BenDecoder, BenFrame};
use ben::utils::rle_to_vec;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::ParquetCompression;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, Cursor};
use std::path::Path;

/// Parquet compression to write with. Snappy is fast and plenty compact for
/// tally outputs; Brotli only pays off when storage is the bottleneck.
pub fn parquet_compression(high: bool) -> ParquetCompression {
    if high {
        ParquetCompression::Brotli(None)
    } else {
        ParquetCompression::Snappy
    }
}

const BATCH: usize = 256;

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

fn process_batch<Row, P>(process: &P, frames: &[BenFrame]) -> Vec<(u16, Row)>
where
    Row: Send,
    P: Fn(&[u16], u16) -> Row + Sync,
{
    frames
        .par_iter()
        .map(|frame| {
            let assignment = decode_frame(frame);
            let row = process(&assignment, frame.count);
            (frame.count, row)
        })
        .collect()
}

/// Walk the BEN file once (fast — just counts frames) and return the total
/// sample count. Useful for Vec preallocation before calling `run_pipeline`.
pub fn count_samples(in_file: &str) -> io::Result<usize> {
    count_samples_from_file(Path::new(in_file), "ben")
}

/// Run a per-sample `process` closure over every record in `in_file`, with
/// frame extraction serial on the caller thread and RLE-decode + `process`
/// fused in parallel across a rayon pool.
///
/// `on_row` is called in BEN-file order with
/// `(sample_count, n_reps, accepted_count, row)` — `sample_count` advances
/// by `n_reps` (MkvChain frames can carry >1), `accepted_count` by 1.
///
/// `process` takes `(&[u16], u16)` = `(assignment, n_reps)` and is
/// invoked inside the rayon pool; it must be `Sync` and produce `Send` rows.
pub fn run_pipeline<Row, P, F>(
    in_file: &str,
    total_samples: usize,
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
        for (n_reps, row) in process_batch(&process, &frame_batch) {
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
        for (n_reps, row) in process_batch(&process, &frame_batch) {
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
