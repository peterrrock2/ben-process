use crate::cli::build_output_path;
use ben::decode::{count_samples_from_file, BenDecoder};
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

    // Frame-only walk — much cheaper than iterating the decoder just to count.
    let total_samples = count_samples_from_file(Path::new(in_ben_file), "ben")
        .expect("Failed to count samples in BEN file");
    eprintln!("Found {} unique plans in {:?}", total_samples, basename);

    let line_count = max_accepted.unwrap_or(total_samples);

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
                // Apply the accumulated permutation in-place.
                for v in assignment.iter_mut() {
                    *v = current_permutation[*v as usize];
                }

                if with_random_reassignments && rng.random_bool(0.5) {
                    let (_idx, (a, b)) =
                        find_first_disagreement_index(&curr_assignment, &assignment)
                            .unwrap_or((0, (1, 1)));
                    swap_labels(&mut assignment, a, b);
                    swap_labels(&mut current_permutation, a, b);
                }

                // Update per-node flip count.
                for ((c, a), d) in curr_assignment
                    .iter()
                    .zip(assignment.iter())
                    .zip(dif_count.iter_mut())
                {
                    if c != a {
                        *d += 1;
                    }
                }
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

    let final_count: Vec<f64> = if normalize {
        dif_count
            .iter()
            .map(|&x| x as f64 / (line_count - 1) as f64)
            .collect()
    } else {
        dif_count.iter().map(|&x| x as f64).collect()
    };

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
