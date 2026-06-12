//! Label-canonical hashing for assignment vectors.
//!
//! Two assignments that differ only by a permutation of district labels describe the same
//! partition. [`canonical_hash`] relabels districts in order of first appearance and hashes the
//! canonicalized sequence with xxh3-128, so equivalent partitions collide on a single 128-bit
//! digest.

use xxhash_rust::xxh3::Xxh3;

/// Hash an assignment vector by its partition (label-invariant).
pub fn canonical_hash(assignment: &[u16]) -> u128 {
    let max_label = assignment.iter().copied().max().unwrap_or(0) as usize;
    // u16::MAX is the "not yet seen" sentinel — assignments using all u16 labels would already
    // overflow the canonical id space.
    let mut remap: Vec<u16> = vec![u16::MAX; max_label + 1];
    let mut next_id: u16 = 0;

    let mut hasher = Xxh3::new();
    for &label in assignment {
        let label_index = label as usize;
        let canonical = if remap[label_index] == u16::MAX {
            let id = next_id;
            remap[label_index] = id;
            next_id += 1;
            id
        } else {
            remap[label_index]
        };
        hasher.update(&canonical.to_le_bytes());
    }
    hasher.digest128()
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
            canonical_hash(&[1, 1, 2, 2, 3, 3]),
            canonical_hash(&[7, 7, 9, 9, 4, 4])
        );
        assert_eq!(
            canonical_hash(&[1, 2, 2, 3, 3, 1]),
            canonical_hash(&[4, 9, 9, 7, 7, 4])
        );
    }

    #[test]
    fn canonical_hash_handles_empty_assignment() {
        // `max().unwrap_or(0)` on an empty slice yields 0 → remap of length 1 with no labels ever
        // inserted. The loop runs zero times, so the digest is the empty xxh3 hash. Pin that the
        // call doesn't panic and that two empty assignments collide (label-invariance trivially
        // holds).
        let h1 = canonical_hash(&[]);
        let h2 = canonical_hash(&[]);
        assert_eq!(h1, h2);
    }

    #[test]
    fn canonical_hash_handles_label_at_u16_max() {
        // u16::MAX is used internally as the "not yet remapped" sentinel, but since `remap` is
        // initialized to that sentinel everywhere, a label value of u16::MAX is correctly detected
        // as first-seen on its first appearance. Pin both the no-panic path and label-invariance
        // with a u16::MAX label present.
        let plan = vec![u16::MAX, 0, u16::MAX, 0];
        let relabeled = vec![7u16, 9, 7, 9];
        assert_eq!(canonical_hash(&plan), canonical_hash(&relabeled));
        assert_ne!(
            canonical_hash(&plan),
            canonical_hash(&[u16::MAX, u16::MAX, 0, 0])
        );
    }

    #[test]
    fn canonical_hash_distinguishes_different_partitions() {
        assert_ne!(canonical_hash(&[1, 1, 2, 2]), canonical_hash(&[1, 2, 1, 2]));
        assert_ne!(canonical_hash(&[1, 1, 2, 2]), canonical_hash(&[1, 2, 2, 1]));
    }

    #[test]
    fn canonical_hash_matches_for_random_relabelings() {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE);

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
        let mut rng = StdRng::seed_from_u64(0xBAD5EED);

        for _ in 0..200 {
            let plan = random_binary_plan_with_both_labels(&mut rng, 24);
            let mut changed = plan.clone();
            let flip_index = rng.random_range(0..changed.len());
            changed[flip_index] = if changed[flip_index] == 1 { 2 } else { 1 };

            assert_ne!(plan, changed);
            assert_ne!(canonical_hash(&plan), canonical_hash(&changed));
        }
    }
}
