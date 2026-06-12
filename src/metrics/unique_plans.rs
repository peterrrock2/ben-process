//! Count distinct *partitions* (label-invariant) in a BEN file.
//!
//! Each accepted frame's assignment is canonicalized by relabeling districts in order of first
//! appearance, then hashed with xxh3-128. Plans that differ only by a permutation of district
//! labels collide on the same digest and thus count as the same partition.

use crate::metrics::canonical::canonical_hash;
use crate::output::parquet::write_unique_plans;
use crate::pipeline::{parquet_compression, run_label_invariant_pipeline, AssignmentLengthCheck};
use std::collections::HashSet;
use std::fs::File;

pub fn count_and_save_unique_plans(
    in_file_name: &str,
    out_file_name: &str,
    show_progress: bool,
    high_compression: bool,
) -> crate::error::Result<()> {
    let mut unique: HashSet<u128> = HashSet::new();
    let mut total_frames: u64 = 0;

    run_label_invariant_pipeline(
        in_file_name,
        // No graph is loaded for this mode — there is no node count to validate against, so the
        // pipeline establishes the expected length from the first frame and a corrupt
        // mixed-length file errors instead of counting the odd frames as distinct plans. The
        // partition is label-invariant by design, hence this entry point: no fixed district-set
        // check applies.
        AssignmentLengthCheck::UniformWithinFile,
        |assignment, _n_reps| Ok(canonical_hash(assignment)),
        |_step, _n_reps, _accepted, hash| {
            unique.insert(hash);
            total_frames += 1;
            Ok(())
        },
        show_progress,
    )?;

    let n_unique = unique.len() as u64;
    log::info!(
        "Unique plans: {} (out of {} accepted frames)",
        n_unique,
        total_frames
    );

    let out = File::create(out_file_name)?;
    write_unique_plans(
        out,
        n_unique,
        total_frames,
        parquet_compression(high_compression),
    )?;

    log::info!("Wrote {}", out_file_name);
    Ok(())
}
