//! Count distinct *partitions* (label-invariant) in a BEN file.
//!
//! Each accepted frame's assignment is canonicalized by relabeling districts in order of first
//! appearance, then hashed with xxh3-128. Plans that differ only by a permutation of district
//! labels collide on the same digest and thus count as the same partition.

use crate::metrics::canonical::canonical_hash;
use crate::pipeline::run_pipeline;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

pub fn count_and_save_unique_plans(
    in_file_name: &str,
    out_file_name: &str,
    show_progress: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut unique: HashSet<u128> = HashSet::new();
    let mut total_frames: u64 = 0;

    run_pipeline(
        in_file_name,
        // No graph is loaded for this mode — it only hashes the raw assignment, so there is no
        // node count to validate against. The partition is label-invariant by design, so
        // the fixed district-set check is deliberately disabled (`None`) too.
        None,
        None,
        |assignment, _n_reps| Ok((0u128, canonical_hash(assignment))),
        |_step, _n_reps, _accepted, hash| {
            unique.insert(hash);
            total_frames += 1;
            Ok(())
        },
        show_progress,
    )?;

    let n_unique = unique.len();
    log::info!(
        "Unique plans: {} (out of {} accepted frames)",
        n_unique,
        total_frames
    );

    let mut out = File::create(out_file_name)?;
    writeln!(out, "unique_plans: {}", n_unique)?;
    writeln!(out, "total_accepted_frames: {}", total_frames)?;

    log::info!("Wrote {}", out_file_name);
    Ok(())
}
