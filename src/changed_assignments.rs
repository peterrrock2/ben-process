use crate::cli::build_output_path;
use ben::decode::BenDecoder;
use pbr::ProgressBar;
use rand::RngExt;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

fn find_first_disagreement_index(vec1: &[u16], vec2: &[u16]) -> Option<(usize, (u16, u16))> {
    vec1.iter()
        .zip(vec2.iter())
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, (&a, &b))| (i, (a, b)))
}

/// Tallies and saves the number of changed assignments (flips) to a text file.
///
/// # Arguments
///
/// * `in_ben_file` - A string slice that holds the path to the BEN file to read from.
/// * `out_file_name` - A string slice that holds the path to the text file to save to.
/// * `normalize` - A flag on whether to normalize the results relative to the number
///     of possible times that a partition could be flipped (the normalization will be
///     on the scale of [0.0, 0.5] due to the way that reassignment works).
/// * `max_accepted` - An optional flag on the maximum number of accepted changes to
///     consider. If `None`, all changes will be considered.
/// * `with_random_reassignments` - A flag to determine if the random reassignments should
///     be used when considering a merge-split operation for ensembles arising from a
///     MCMC method. The code fore many of these methods has an inherit bias towards a
///     particular way of labeling the districts which can bias the change-assignment count
///     since it may favor canonicalizing the assignment or take the convention that the
///     district with the most moved population gets the smaller label. To account for
///     these choices and to reconstruct the ensemble appropriately, we need to keep track
///     of the merged and split districts and then randomize the reassignment labels. Do not
///     set this flag to true if using a method that does not use MCMC merge-split.
///
/// # Returns
///
/// * `std::result::Result<(), Box<dyn std::error::Error>>` - A result containing the success or
///     failure of the operation.
pub fn tally_and_save_changed_assignments(
    in_ben_file: &str,
    normalize: bool,
    max_accepted: Option<usize>,
    with_random_reassignments: bool,
    show_progress: bool,
    output_dir: Option<&str>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut ben_file = File::open(in_ben_file).expect("BEN file not found");
    let mut rng = rand::rng();

    let line_checker = BenDecoder::new(&ben_file).expect("Failed to initialize decoder");

    let basename = Path::new(in_ben_file)
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy();

    eprintln!("Reading {:?}...", basename);

    let mut line_count: usize = 0;
    for _ in line_checker.enumerate() {
        line_count += 1;
    }

    eprintln!("Found {:?} unique plans in {:?}\r", line_count, basename);

    if let Some(max_accepted) = max_accepted {
        line_count = max_accepted as usize;
    }

    let out_file_name = build_output_path(
        in_ben_file,
        format!("_accept_{}_changed_assignments.txt", line_count).as_str(),
        output_dir,
    );

    let mut n_pb_tics = 100;

    let mut pb_step_size = (line_count / n_pb_tics as usize) as usize;

    if line_count < n_pb_tics as usize {
        n_pb_tics = line_count as u64;
        pb_step_size = 1;
    }

    let mut pb = if show_progress {
        Some(ProgressBar::new(n_pb_tics))
    } else {
        None
    };

    ben_file.seek(SeekFrom::Start(0))?;

    let ben_reader = std::io::BufReader::new(ben_file);

    let mut decoder = match BenDecoder::new(ben_reader) {
        Ok(decoder) => decoder,
        Err(e) => {
            eprintln!("Failed to initialize BenDecoder: {:?}", e);
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Decoder initialization failed",
            )));
        }
    };

    let mut out = File::create(&out_file_name)
        .expect("Could not create output file. The file may already exist.");

    let (mut curr_assignment, mut dif_count) = if let Some(result) = decoder.next() {
        match result {
            Ok((assignment, _)) => {
                (assignment.clone(), vec![0; assignment.len()]) // Return as a tuple
            }
            Err(e) => {
                eprintln!("Error decoding sample: {:?}", e);
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Decoding failed",
                )));
            }
        }
    } else {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "No data found",
        )));
    };

    let mut count: usize = 1;
    let mut full_count: usize = 1;
    let max_assignment = *curr_assignment.iter().max().unwrap();
    let mut current_permutation = (0..=max_assignment).collect::<Vec<u16>>();
    for result in decoder {
        count += 1;
        full_count += 1;
        match result {
            Ok((mut assignment, _)) => {
                // NOTE: the current assignment will already have the permutation
                // applied since it is the assignment from the previous iteration of
                // the loop.
                assignment = assignment
                    .iter_mut()
                    .map(|&mut v| current_permutation[v as usize])
                    .collect::<Vec<u16>>();
                if with_random_reassignments {
                    // Flip the assignment with probablitly 0.5
                    if rng.random_bool(0.5) {
                        let (_idx, (a, b)) =
                            find_first_disagreement_index(&curr_assignment, &assignment)
                                .unwrap_or_else(|| (0, (1, 1)));
                        assignment = assignment
                            .iter_mut()
                            .map(|&mut v| {
                                if v == a {
                                    b
                                } else if v == b {
                                    a
                                } else {
                                    v
                                }
                            })
                            .collect::<Vec<u16>>();
                        current_permutation = current_permutation
                            .iter()
                            .map(|&v| {
                                if v == a {
                                    b
                                } else if v == b {
                                    a
                                } else {
                                    v
                                }
                            })
                            .collect::<Vec<u16>>()
                    }
                }
                curr_assignment
                    .iter()
                    .zip(assignment.iter().zip(dif_count.iter_mut()))
                    .for_each(|(a, (b, c))| {
                        if a != b {
                            *c += 1;
                        }
                    });
                curr_assignment = assignment;
            }
            Err(e) => {
                eprintln!("Error decoding sample: {:?}", e);
                break;
            }
        }
        if show_progress && count > pb_step_size {
            pb.as_mut().unwrap().inc();
            count = count - pb_step_size;
        }
        if full_count >= line_count {
            break;
        }
    }

    // NOTE: We divide by line_count - 1 because if there are n accpeted steps
    // then we can reassign a single unit at most n - 1 times.
    let final_count = if normalize {
        dif_count
            .iter()
            .map(|&x| x as f64 / (line_count - 1) as f64)
            .collect::<Vec<f64>>()
    } else {
        dif_count.iter().map(|&x| x as f64).collect::<Vec<f64>>()
    };

    if let Some(pb_ref) = pb.as_mut() {
        pb_ref.finish();
    }
    eprintln!("Final count: {:?}", full_count);
    eprintln!("Writing final output...");

    out.write(format!("{:?}", final_count).as_bytes())
        .expect("Could not write to output file");
    out.write(format!("\nTotal Accepted: {:?}", line_count).as_bytes())
        .expect("Could not write to output file");

    eprintln!("Done!");
    Ok(())
}
