use crate::cli::build_output_path;
use crate::district::{observed_assignment_districts, validate_district_set_unchanged};
use crate::error::BenError;
use crate::output::parquet::write_changed_assignments;
use crate::pipeline::{count_frames, parquet_compression, run_sequential_accepted_frames};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::fs::File;
use std::io;
use std::path::Path;

fn find_first_disagreement_index(first: &[u16], second: &[u16]) -> Option<(usize, (u16, u16))> {
    first
        .iter()
        .zip(second.iter())
        .enumerate()
        .find(|(_, (label_a, label_b))| label_a != label_b)
        .map(|(index, (&label_a, &label_b))| (index, (label_a, label_b)))
}

/// Swap all occurrences of `label_a` with `label_b` (and vice versa) in `labels`, in place.
fn swap_labels(labels: &mut [u16], label_a: u16, label_b: u16) {
    for label in labels.iter_mut() {
        if *label == label_a {
            *label = label_b;
        } else if *label == label_b {
            *label = label_a;
        }
    }
}

fn validate_labels_within_first_assignment_range(
    assignment: &[u16],
    current_permutation: &[u16],
) -> io::Result<()> {
    let max_label = current_permutation
        .len()
        .checked_sub(1)
        .expect("current_permutation should always contain at least label 0");

    for (index, &label) in assignment.iter().enumerate() {
        if label as usize > max_label {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "encountered assignment label {} at index {} outside first assignment label \
                    range 0..={}; changed-assignments cannot apply the current label permutation",
                    label, index, max_label
                ),
            ));
        }
    }

    Ok(())
}

fn update_changed_assignment_state(
    current_assignment: &[u16],
    assignment: &mut [u16],
    current_permutation: &mut [u16],
    diff_count: &mut [u32],
    randomize_reassignment: bool,
) -> io::Result<()> {
    validate_labels_within_first_assignment_range(assignment, current_permutation)?;

    for label in assignment.iter_mut() {
        *label = current_permutation[*label as usize];
    }

    if randomize_reassignment {
        let (_index, (label_a, label_b)) =
            find_first_disagreement_index(current_assignment, assignment).unwrap_or((0, (1, 1)));
        swap_labels(assignment, label_a, label_b);
        swap_labels(current_permutation, label_a, label_b);
    }

    for ((current_label, updated_label), diff) in current_assignment
        .iter()
        .zip(assignment.iter())
        .zip(diff_count.iter_mut())
    {
        if current_label != updated_label {
            *diff += 1;
        }
    }

    Ok(())
}

/// Pure driver loop over an iterator of assignments. Takes `first_assignment` as the seed and
/// consumes `rest` to produce the raw `diff_count`. The `should_randomize` closure is invoked once
/// per non-seed frame to decide whether that step's update applies a label-swap reassignment —
/// separating the RNG (or the test's fake RNG) from this loop so callers can drive it
/// deterministically.
///
/// Returns `(diff_count, full_count)` where `full_count` includes the seed frame plus everything
/// yielded by `rest`.
#[cfg(test)]
fn compute_changed_counts<I, F>(
    first_assignment: &[u16],
    rest: I,
    mut should_randomize: F,
) -> (Vec<u32>, usize)
where
    I: IntoIterator<Item = Vec<u16>>,
    F: FnMut() -> bool,
{
    let n = first_assignment.len();
    let mut diff_count = vec![0u32; n];
    let max_assignment = *first_assignment.iter().max().unwrap_or(&0);
    let mut current_permutation: Vec<u16> = (0..=max_assignment).collect();
    let mut current_assignment = first_assignment.to_vec();
    let mut full_count: usize = 1;

    for mut assignment in rest {
        full_count += 1;
        update_changed_assignment_state(
            &current_assignment,
            &mut assignment,
            &mut current_permutation,
            &mut diff_count,
            should_randomize(),
        )
        .expect("test assignments should stay within the first assignment label range");
        current_assignment = assignment;
    }

    (diff_count, full_count)
}

fn finalize_changed_counts(diff_count: &[u32], line_count: usize, normalize: bool) -> Vec<f64> {
    if !normalize {
        return diff_count.iter().map(|&x| x as f64).collect();
    }

    if line_count <= 1 {
        return vec![0.0; diff_count.len()];
    }

    diff_count
        .iter()
        .map(|&x| x as f64 / (line_count - 1) as f64)
        .collect()
}

/// Tallies the number of changed assignments (flips) per node and saves them to a Parquet file with
/// columns `node` and `changed_assignments`, one row per node.
///
/// # Arguments
///
/// * `in_ben_file` - A string slice that holds the path to the BEN file to read from.
/// * `normalize` - Whether to normalize flip counts by `line_count - 1` (max possible flips per
///   unit).
/// * `max_accepted` - Optional cap on the number of accepted changes considered.
/// * `with_random_reassignments` - Randomize merge-split label reassignments. Set only for MCMC
///   merge-split ensembles; off otherwise.
/// * `seed` - Optional seed for the reassignment RNG; when `None`, an OS-seeded RNG is used and the
///   randomized run is not reproducible. Ignored unless `with_random_reassignments` is set.
/// * `show_progress` - Draw an indicatif progress bar on stderr.
/// * `output_dir` - Optional directory for the output file.
/// * `high_compression` - Use Brotli instead of Snappy for the Parquet output.
#[allow(clippy::too_many_arguments)]
pub fn tally_and_save_changed_assignments(
    in_ben_file: &str,
    normalize: bool,
    max_accepted: Option<usize>,
    with_random_reassignments: bool,
    seed: Option<u64>,
    show_progress: bool,
    output_dir: Option<&str>,
    high_compression: bool,
) -> crate::error::Result<()> {
    // A seed makes `--randomize-reassignments` reproducible; without one we seed from the
    // OS-backed thread RNG.
    let mut rng = match seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_rng(&mut rand::rng()),
    };

    let basename = Path::new(in_ben_file)
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    log::info!("Reading {:?}...", basename);

    // Changed-assignments works per *accepted record* (frame), not per repeated sample. For
    // MkvChain BEN files a frame can carry a repetition count > 1 — those repeats represent the
    // SAME assignment and therefore zero flips among themselves. So we count frames, not samples.
    let total_frames = count_frames(in_ben_file)?;
    log::info!("Found {} accepted plans in {:?}", total_frames, basename);

    // A `--max-accepted` beyond the file's frame count means "everything": clamp it so the
    // normalization divisor below and the `_accept_N_` output filename both reflect the frames
    // actually consumed, not the requested cap.
    let line_count = max_accepted.map_or(total_frames, |cap| cap.min(total_frames));

    let out_file_name = build_output_path(
        in_ben_file,
        format!("_accept_{}_changed_assignments.parquet", line_count).as_str(),
        output_dir,
    );

    let out = File::create(&out_file_name).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("could not create changed-assignments output file {out_file_name:?}: {e}"),
        )
    })?;

    let mut current_assignment: Option<Vec<u16>> = None;
    let mut current_permutation: Vec<u16> = Vec::new();
    let mut diff_count: Vec<u32> = Vec::new();
    // The first frame's district label set; every later frame must use the same set, matching the
    // fixed-district-set invariant the run_pipeline modes enforce centrally.
    let mut expected_district_set: Option<u128> = None;

    let full_count = run_sequential_accepted_frames(
        in_ben_file,
        total_frames,
        Some(line_count),
        show_progress,
        |frame| {
            let observed = observed_assignment_districts(&frame.assignment)?.1;
            match expected_district_set {
                None => expected_district_set = Some(observed),
                Some(expected) => {
                    validate_district_set_unchanged(observed, expected, "changed-assignments")?
                }
            }

            if current_assignment.is_none() {
                let max_assignment = *frame.assignment.iter().max().unwrap_or(&0);
                current_permutation = (0..=max_assignment).collect();
                diff_count = vec![0u32; frame.assignment.len()];
                current_assignment = Some(frame.assignment);
                return Ok(());
            }

            let mut assignment = frame.assignment;
            update_changed_assignment_state(
                current_assignment
                    .as_ref()
                    .expect("current assignment should be initialized after first frame"),
                &mut assignment,
                &mut current_permutation,
                &mut diff_count,
                with_random_reassignments && rng.random_bool(0.5),
            )?;
            current_assignment = Some(assignment);
            Ok(())
        },
    )?;

    if full_count == 0 {
        return Err(BenError::NoData);
    }

    let final_count = finalize_changed_counts(&diff_count, line_count, normalize);
    log::info!("Final count: {}", full_count);
    log::info!("Writing final output...");

    write_changed_assignments(out, &final_count, parquet_compression(high_compression))?;

    log::info!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        compute_changed_counts, finalize_changed_counts, find_first_disagreement_index,
        swap_labels, update_changed_assignment_state,
        validate_labels_within_first_assignment_range,
    };

    #[test]
    fn disagreement_index_returns_first_mismatch() {
        assert_eq!(
            find_first_disagreement_index(&[1, 2, 3], &[1, 9, 3]),
            Some((1, (2, 9)))
        );
        assert_eq!(find_first_disagreement_index(&[1, 2], &[1, 2]), None);
    }

    #[test]
    fn swap_labels_rewrites_all_occurrences() {
        let mut labels = vec![1, 2, 3, 2, 1];
        swap_labels(&mut labels, 1, 2);
        assert_eq!(labels, vec![2, 1, 3, 1, 2]);
    }

    #[test]
    fn update_state_applies_existing_permutation_before_counting() {
        let current_assignment = vec![1, 1, 2, 2];
        let mut next_assignment = vec![2, 2, 1, 1];
        let mut current_permutation = vec![0, 2, 1];
        let mut diff_count = vec![0u32; 4];

        update_changed_assignment_state(
            &current_assignment,
            &mut next_assignment,
            &mut current_permutation,
            &mut diff_count,
            false,
        )
        .unwrap();

        assert_eq!(next_assignment, current_assignment);
        assert_eq!(diff_count, vec![0, 0, 0, 0]);
        assert_eq!(current_permutation, vec![0, 2, 1]);
    }

    #[test]
    fn update_state_randomized_reassignment_swaps_labels_and_updates_permutation() {
        let current_assignment = vec![1, 1, 2, 2];
        let mut next_assignment = vec![1, 2, 1, 2];
        let mut current_permutation = vec![0, 1, 2];
        let mut diff_count = vec![0u32; 4];

        update_changed_assignment_state(
            &current_assignment,
            &mut next_assignment,
            &mut current_permutation,
            &mut diff_count,
            true,
        )
        .unwrap();

        assert_eq!(next_assignment, vec![2, 1, 2, 1]);
        assert_eq!(current_permutation, vec![0, 2, 1]);
        assert_eq!(diff_count, vec![1, 0, 0, 1]);
    }

    #[test]
    fn validates_later_assignment_labels_before_permutation_indexing() {
        let err = validate_labels_within_first_assignment_range(&[1, 3], &[0, 1, 2]).unwrap_err();
        assert!(
            err.to_string().contains(
                "encountered assignment label 3 at index 1 outside first assignment label range 0..=2"
            ),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn compute_changed_counts_with_no_randomization_matches_pure_diff() {
        // Three accepted plans; randomization disabled, so diff_count is just the elementwise count
        // of how many transitions changed each unit.
        let first = vec![1u16, 1, 2, 2];
        let rest = vec![vec![1u16, 2, 1, 2], vec![2u16, 2, 1, 1]];

        let (diff_count, full_count) = compute_changed_counts(&first, rest, || false);

        // p0 -> p1: differ at idx 1, 2 -> [0, 1, 1, 0] p1 -> p2: differ at idx 0, 3 -> [1, 1, 1, 1]
        assert_eq!(diff_count, vec![1, 1, 1, 1]);
        assert_eq!(full_count, 3);
    }

    #[test]
    fn compute_changed_counts_randomize_always_swaps_first_disagreement() {
        // With should_randomize() == true on every step, the inner state updater swaps labels at
        // the first disagreement before counting, which collapses pure label permutations to zero
        // diffs.
        let first = vec![1u16, 1, 2, 2];
        // Pure relabeling of `first`: permuting 1<->2 should produce zero diffs once the swap is
        // applied.
        let rest = vec![vec![2u16, 2, 1, 1]];

        let (diff_count, full_count) = compute_changed_counts(&first, rest, || true);

        assert_eq!(diff_count, vec![0, 0, 0, 0]);
        assert_eq!(full_count, 2);
    }

    #[test]
    fn compute_changed_counts_invokes_randomize_callback_per_step() {
        // The closure must be called exactly once per non-seed frame, and the per-step decision
        // must be honoured. Program a sequence and assert it's fully consumed in order.
        let first = vec![1u16, 1, 2, 2];
        let rest = vec![
            vec![1u16, 2, 1, 2],
            vec![2u16, 2, 1, 1],
            vec![1u16, 1, 1, 2],
        ];

        let mut decisions = vec![false, true, false].into_iter();
        let (_diff_count, full_count) =
            compute_changed_counts(&first, rest, || decisions.next().unwrap());

        assert_eq!(full_count, 4);
        assert!(decisions.next().is_none(), "callback called too few times");
    }

    #[test]
    fn finalize_counts_normalizes_single_plan_to_zeroes() {
        assert_eq!(
            finalize_changed_counts(&[0, 0, 0], 1, true),
            vec![0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn finalize_counts_divides_by_transition_count_when_normalizing() {
        assert_eq!(
            finalize_changed_counts(&[0, 2, 1], 3, true),
            vec![0.0, 1.0, 0.5]
        );
        assert_eq!(
            finalize_changed_counts(&[0, 2, 1], 3, false),
            vec![0.0, 2.0, 1.0]
        );
    }
}
