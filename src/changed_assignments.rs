use crate::cli::build_output_path;
use crate::pipeline::count_frames;
use ben::decode::BenDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use rand::RngExt;
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::Path;

fn find_first_disagreement_index(vec1: &[u16], vec2: &[u16]) -> Option<(usize, (u16, u16))> {
    vec1.iter()
        .zip(vec2.iter())
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (&a, &b))| (i, (a, b)))
}

/// Swap all occurrences of `a` with `b` (and vice versa) in `v`, in place.
fn swap_labels(v: &mut [u16], a: u16, b: u16) {
    for x in v.iter_mut() {
        if *x == a {
            *x = b;
        } else if *x == b {
            *x = a;
        }
    }
}

fn update_changed_assignment_state(
    curr_assignment: &[u16],
    assignment: &mut [u16],
    current_permutation: &mut [u16],
    dif_count: &mut [u32],
    randomize_reassignment: bool,
) {
    for v in assignment.iter_mut() {
        *v = current_permutation[*v as usize];
    }

    if randomize_reassignment {
        let (_idx, (a, b)) = find_first_disagreement_index(curr_assignment, assignment)
            .unwrap_or((0, (1, 1)));
        swap_labels(assignment, a, b);
        swap_labels(current_permutation, a, b);
    }

    for ((c, a), d) in curr_assignment
        .iter()
        .zip(assignment.iter())
        .zip(dif_count.iter_mut())
    {
        if c != a {
            *d += 1;
        }
    }
}

fn finalize_changed_counts(dif_count: &[u32], line_count: usize, normalize: bool) -> Vec<f64> {
    if !normalize {
        return dif_count.iter().map(|&x| x as f64).collect();
    }

    if line_count <= 1 {
        return vec![0.0; dif_count.len()];
    }

    dif_count
        .iter()
        .map(|&x| x as f64 / (line_count - 1) as f64)
        .collect()
}

/// Tallies and saves the number of changed assignments (flips) to a text file.
///
/// # Arguments
///
/// * `in_ben_file` - A string slice that holds the path to the BEN file to read from.
/// * `normalize` - Whether to normalize flip counts by `line_count - 1` (max
///   possible flips per unit).
/// * `max_accepted` - Optional cap on the number of accepted changes considered.
/// * `with_random_reassignments` - Randomize merge-split label reassignments.
///   Set only for MCMC merge-split ensembles; off otherwise.
/// * `show_progress` - Draw an indicatif progress bar on stderr.
/// * `output_dir` - Optional directory for the output file.
pub fn tally_and_save_changed_assignments(
    in_ben_file: &str,
    normalize: bool,
    max_accepted: Option<usize>,
    with_random_reassignments: bool,
    show_progress: bool,
    output_dir: Option<&str>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let ben_file = File::open(in_ben_file).expect("BEN file not found");
    let mut rng = rand::rng();

    let basename = Path::new(in_ben_file)
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy()
        .into_owned();
    eprintln!("Reading {:?}...", basename);

    // Changed-assignments works per *accepted record* (frame), not per
    // repeated sample. For MkvChain BEN files a frame can carry a repetition
    // count > 1 — those repeats represent the SAME assignment and therefore
    // zero flips among themselves. So we count frames, not samples.
    let total_frames = count_frames(in_ben_file).expect("Failed to count frames in BEN file");
    eprintln!("Found {} accepted plans in {:?}", total_frames, basename);

    let line_count = max_accepted.unwrap_or(total_frames);

    let out_file_name = build_output_path(
        in_ben_file,
        format!("_accept_{}_changed_assignments.txt", line_count).as_str(),
        output_dir,
    );

    let pb = if show_progress {
        let pb = ProgressBar::new(line_count as u64);
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

    let ben_reader = BufReader::new(ben_file);
    let mut decoder = BenDecoder::new(ben_reader).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Failed to initialize BenDecoder: {:?}", e),
        )
    })?;

    let mut out = File::create(&out_file_name)
        .expect("Could not create output file. The file may already exist.");

    // First plan seeds curr_assignment and the zero dif_count.
    let (mut curr_assignment, mut dif_count) = match decoder.next() {
        Some(Ok((assignment, _))) => {
            let n = assignment.len();
            (assignment, vec![0u32; n])
        }
        Some(Err(e)) => return Err(Box::new(e)),
        None => {
            return Err(Box::new(io::Error::new(
                io::ErrorKind::Other,
                "No data found",
            )))
        }
    };
    if let Some(pb) = &pb {
        pb.inc(1);
    }

    let max_assignment = *curr_assignment.iter().max().unwrap_or(&0);
    let mut current_permutation: Vec<u16> = (0..=max_assignment).collect();

    let mut full_count: usize = 1;
    for result in decoder {
        full_count += 1;
        match result {
            Ok((mut assignment, _)) => {
                let randomize_reassignment =
                    with_random_reassignments && rng.random_bool(0.5);
                update_changed_assignment_state(
                    &curr_assignment,
                    &mut assignment,
                    &mut current_permutation,
                    &mut dif_count,
                    randomize_reassignment,
                );
                curr_assignment = assignment;
            }
            Err(e) => {
                eprintln!("Error decoding sample: {:?}", e);
                break;
            }
        }
        if let Some(pb) = &pb {
            pb.inc(1);
        }
        if full_count >= line_count {
            break;
        }
    }

    let final_count = finalize_changed_counts(&dif_count, line_count, normalize);

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }
    eprintln!("Final count: {}", full_count);
    eprintln!("Writing final output...");

    out.write_all(format!("{:?}", final_count).as_bytes())?;
    out.write_all(format!("\nTotal Accepted: {:?}", line_count).as_bytes())?;

    eprintln!("Done!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        finalize_changed_counts, find_first_disagreement_index, swap_labels,
        update_changed_assignment_state,
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
        let curr_assignment = vec![1, 1, 2, 2];
        let mut next_assignment = vec![2, 2, 1, 1];
        let mut current_permutation = vec![0, 2, 1];
        let mut dif_count = vec![0u32; 4];

        update_changed_assignment_state(
            &curr_assignment,
            &mut next_assignment,
            &mut current_permutation,
            &mut dif_count,
            false,
        );

        assert_eq!(next_assignment, curr_assignment);
        assert_eq!(dif_count, vec![0, 0, 0, 0]);
        assert_eq!(current_permutation, vec![0, 2, 1]);
    }

    #[test]
    fn update_state_randomized_reassignment_swaps_labels_and_updates_permutation() {
        let curr_assignment = vec![1, 1, 2, 2];
        let mut next_assignment = vec![1, 2, 1, 2];
        let mut current_permutation = vec![0, 1, 2];
        let mut dif_count = vec![0u32; 4];

        update_changed_assignment_state(
            &curr_assignment,
            &mut next_assignment,
            &mut current_permutation,
            &mut dif_count,
            true,
        );

        assert_eq!(next_assignment, vec![2, 1, 2, 1]);
        assert_eq!(current_permutation, vec![0, 2, 1]);
        assert_eq!(dif_count, vec![1, 0, 0, 1]);
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
