//! Extract distinct *partitions* (label-invariant) and re-encode them as
//! a Standard BEN file. Dedup key is xxh3-128 of the canonical relabeling
//! (districts numbered in order of first appearance); the assignment that
//! gets written is the **first-seen original**, so labels in the output
//! match how the plan first appeared in the input.

use crate::pipeline::count_frames;
use ben::decode::BenDecoder;
use ben::encode::BenEncoder;
use ben::BenVariant;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use xxhash_rust::xxh3::Xxh3;

fn canonical_hash(assignment: &[u16]) -> u128 {
    let max_label = assignment.iter().copied().max().unwrap_or(0) as usize;
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

pub fn extract_unique_plans(
    in_file_name: &str,
    out_file_name: &str,
    show_progress: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let basename = Path::new(in_file_name)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    eprintln!("Reading {:?}...", basename);

    let total_frames = count_frames(in_file_name)?;
    eprintln!("Found {} accepted plans in {:?}", total_frames, basename);

    let pb = if show_progress {
        let pb = ProgressBar::new(total_frames as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{bar:40.cyan/blue} {pos}/{len} [{elapsed_precise} ETA {eta}]",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        Some(pb)
    } else {
        None
    };

    let in_file = File::open(in_file_name)?;
    let decoder = BenDecoder::new(in_file)?;

    let out_file = File::create(out_file_name)?;
    let mut writer = BufWriter::new(out_file);
    let mut encoder = BenEncoder::new(&mut writer, BenVariant::Standard);

    let mut seen: HashSet<u128> = HashSet::new();
    let mut total: u64 = 0;
    let mut written: u64 = 0;

    for record_res in decoder {
        let (assignment, _n_reps) = record_res?;
        total += 1;
        let h = canonical_hash(&assignment);
        if seen.insert(h) {
            encoder.write_assignment(assignment)?;
            written += 1;
        }
        if let Some(pb) = &pb {
            pb.inc(1);
        }
    }

    encoder.finish()?;
    drop(encoder);

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    eprintln!(
        "Unique plans: {} (out of {} accepted frames)",
        written, total
    );
    eprintln!("Wrote {}", out_file_name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::canonical_hash;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{RngExt, SeedableRng};

    fn random_binary_plan_with_both_labels(rng: &mut StdRng, len: usize) -> Vec<u16> {
        let mut plan: Vec<u16> = (0..len)
            .map(|_| if rng.random_bool(0.5) { 1 } else { 2 })
            .collect();
        if !plan.contains(&1) {
            plan[0] = 1;
        }
        if !plan.contains(&2) {
            plan[0] = 2;
        }
        if !plan.contains(&1) {
            plan[1] = 1;
        }
        plan
    }

    #[test]
    fn canonical_hash_is_invariant_to_label_permutations() {
        assert_eq!(
            canonical_hash(&[1, 2, 2, 3, 3, 1]),
            canonical_hash(&[4, 9, 9, 7, 7, 4])
        );
    }

    #[test]
    fn canonical_hash_changes_when_partition_structure_changes() {
        assert_ne!(canonical_hash(&[1, 1, 2, 2]), canonical_hash(&[1, 2, 2, 1]));
    }

    #[test]
    fn canonical_hash_matches_for_random_relabelings() {
        let mut rng = StdRng::seed_from_u64(0xFACEFEED);

        for _ in 0..200 {
            let plan_len = rng.random_range(8..32);
            let label_count = rng.random_range(2..8);
            let plan: Vec<u16> = (0..plan_len)
                .map(|_| rng.random_range(0..label_count) as u16)
                .collect();

            let mut labels: Vec<u16> = (0..label_count as u16).collect();
            labels.shuffle(&mut rng);
            let relabeled: Vec<u16> = plan.iter().map(|&label| labels[label as usize]).collect();

            assert_eq!(canonical_hash(&plan), canonical_hash(&relabeled));
        }
    }

    #[test]
    fn canonical_hash_differs_for_random_distinct_binary_partitions() {
        let mut rng = StdRng::seed_from_u64(0x1234ABCD);

        for _ in 0..200 {
            let plan = random_binary_plan_with_both_labels(&mut rng, 24);
            let mut changed = plan.clone();
            let idx = rng.random_range(0..changed.len());
            changed[idx] = if changed[idx] == 1 { 2 } else { 1 };

            assert_ne!(plan, changed);
            assert_ne!(canonical_hash(&plan), canonical_hash(&changed));
        }
    }
}
