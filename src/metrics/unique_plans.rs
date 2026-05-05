//! Count distinct *partitions* (label-invariant) in a BEN file.
//!
//! Each accepted frame's assignment is canonicalized by relabeling districts
//! in order of first appearance, then hashed with xxh3-128. Plans that differ
//! only by a permutation of district labels collide on the same digest and
//! thus count as the same partition.

use crate::pipeline::{count_samples, run_pipeline};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use xxhash_rust::xxh3::Xxh3;

fn canonical_hash(assignment: &[u16]) -> u128 {
    let max_label = assignment.iter().copied().max().unwrap_or(0) as usize;
    // u16::MAX is the "not yet seen" sentinel — assignments using all u16
    // labels would already overflow the canonical id space.
    let mut remap: Vec<u16> = vec![u16::MAX; max_label + 1];
    let mut next_id: u16 = 0;

    let mut hasher = Xxh3::new();
    for &label in assignment {
        let idx = label as usize;
        let canonical = if remap[idx] == u16::MAX {
            let id = next_id;
            remap[idx] = id;
            next_id += 1;
            id
        } else {
            remap[idx]
        };
        hasher.update(&canonical.to_le_bytes());
    }
    hasher.digest128()
}

pub fn count_and_save_unique_plans(
    in_file_name: &str,
    out_file_name: &str,
    show_progress: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let total = count_samples(in_file_name)?;

    let mut unique: HashSet<u128> = HashSet::new();
    let mut total_frames: u64 = 0;

    run_pipeline(
        in_file_name,
        total,
        |assignment, _n_reps| canonical_hash(assignment),
        |_step, _n_reps, _accepted, hash| {
            unique.insert(hash);
            total_frames += 1;
        },
        show_progress,
    )?;

    let n_unique = unique.len();
    eprintln!(
        "Unique plans: {} (out of {} accepted frames)",
        n_unique, total_frames
    );

    let mut out = File::create(out_file_name)?;
    writeln!(out, "unique_plans: {}", n_unique)?;
    writeln!(out, "total_accepted_frames: {}", total_frames)?;

    eprintln!("Wrote {}", out_file_name);
    Ok(())
}
