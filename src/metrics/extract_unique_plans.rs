//! Extract distinct *partitions* (label-invariant) and re-encode them as a Standard BEN file. Dedup
//! key is xxh3-128 of the canonical relabeling (districts numbered in order of first appearance);
//! the assignment that gets written is the **first-seen original**, so labels in the output match
//! how the plan first appeared in the input.

use crate::metrics::canonical::canonical_hash;
use crate::pipeline::{count_frames, run_sequential_accepted_frames};
use ben::encode::BenEncoder;
use ben::BenVariant;
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

pub fn extract_unique_plans(
    in_file_name: &str,
    out_file_name: &str,
    show_progress: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let basename = Path::new(in_file_name)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    log::info!("Reading {:?}...", basename);

    let total_frames = count_frames(in_file_name)?;
    log::info!("Found {} accepted plans in {:?}", total_frames, basename);

    let out_file = File::create(out_file_name)?;
    let mut writer = BufWriter::new(out_file);
    let mut encoder = BenEncoder::new(&mut writer, BenVariant::Standard);

    let mut seen: HashSet<u128> = HashSet::new();
    let mut written: u64 = 0;

    let total =
        run_sequential_accepted_frames(in_file_name, total_frames, None, show_progress, |frame| {
            let hash = canonical_hash(&frame.assignment);
            if seen.insert(hash) {
                encoder.write_assignment(frame.assignment)?;
                written += 1;
            }
            Ok(())
        })?;

    encoder.finish()?;
    drop(encoder);

    log::info!(
        "Unique plans: {} (out of {} accepted frames)",
        written,
        total
    );
    log::info!("Wrote {}", out_file_name);
    Ok(())
}
