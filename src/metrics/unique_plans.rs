//! Count distinct *partitions* (label-invariant) in a BEN file.
//!
//! Each accepted frame's assignment is canonicalized by relabeling districts in order of first
//! appearance, then hashed with xxh3-128. Plans that differ only by a permutation of district
//! labels collide on the same digest and thus count as the same partition.

use crate::metrics::canonical::{canonical_hash, validate_assignment_len};
use crate::output::parquet::write_unique_plans;
use crate::pipeline::{parquet_compression, run_pipeline};
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
    // No graph means no graph-length check; the first frame establishes the expected length so a
    // corrupt mixed-length file errors instead of counting the odd frames as distinct plans.
    let mut expected_len: Option<usize> = None;

    run_pipeline(
        in_file_name,
        // No graph is loaded for this mode — it only hashes the raw assignment, so there is no
        // node count to validate against. The partition is label-invariant by design, so
        // the fixed district-set check is deliberately disabled (`None`) too.
        None,
        None,
        |assignment, _n_reps| Ok((0u128, (assignment.len(), canonical_hash(assignment)))),
        |_step, _n_reps, _accepted, (assignment_len, hash)| {
            validate_assignment_len(&mut expected_len, assignment_len)?;
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
