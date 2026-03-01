/// This Rust program processes BEN files and performs various operations based on the specified mode.
/// It supports three modes: TallyKeys, CutEdges, and ChangedAssignments.
///
/// # Modes
///
/// - `TallyKeys`: Tallies and saves data from BEN files to Parquet files based on a list of keys.
/// - `CutEdges`: Tallies and saves the number of cut edges in the graph to Parquet files.
/// - `ChangedAssignments`: Tallies and saves the number of changed assignments to a text file.
///
/// # Arguments
///
/// - `mode`: The mode of operation (TallyKeys, CutEdges, ChangedAssignments).
/// - `graph_file`: The path to the graph file (required for TallyKeys and CutEdges modes).
/// - `ben_file`: The path to the BEN file.
/// - `normalize`: A flag to normalize the results (used in ChangedAssignments mode).
/// - `keys`: A list of keys to tally (used in TallyKeys mode).
///
/// # Structs
///
/// - `Args`: Represents the command-line arguments.
/// - `JsonGraphData`: Represents the structure of the JSON graph data.
/// - `Graph`: Represents the graph with nodes and edges.
///
/// # Functions
///
/// - `main`: The main entry point of the program.
/// - `make_graph_from_json`: Reads and parses the JSON graph file into a `Graph` struct.
/// - `tally_keys`: Tallies the values of specified keys for each partition in the graph.
/// - `save_tallies_to_parquet`: Saves the tallies to a Parquet file.
/// - `tally_and_save_from_key_list`: Tallies and saves data from BEN files based on a list of keys.
/// - `cut_edges`: Counts the number of cut edges in the graph based on the assignment.
/// - `tally_and_save_cut_edges`: Tallies and saves the number of cut edges in the graph to a Parquet file.
/// - `tally_and_save_changed_assignments`: Tallies and saves the number of changed assignments to a text file.
use ben::decode::{count_samples_from_file, BenDecoder};
use clap::{Parser, ValueEnum};
use pbr::ProgressBar;
use polars::prelude::*;
use rand::RngExt;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Result, Value};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Parser, Debug, Clone, ValueEnum, PartialEq)]
enum Mode {
    TallyKeys,
    CutEdges,
    ChangedAssignments,
}

#[derive(Parser, Debug)]
#[command(
    name = "BEN Parquet Tally Tool",
    about = "A tool for tallying and saving data from BEN files to Parquet files.",
    version = "0.1.0"
)]
struct Args {
    #[arg(short, long, default_value = "cut-edges")]
    mode: Mode,
    #[arg(short, long)]
    graph_file: Option<String>,
    #[arg(short, long)]
    ben_file: String,
    #[arg(short, long, default_value_t = false)]
    normalize: bool,
    #[arg(short, long)]
    max_accepted: Option<usize>,
    #[arg(short, long, default_value_t = true)]
    mkv_rand_reassignment_off: bool,
    #[arg(short, long, num_args(1..))]
    keys: Vec<String>,
    #[arg(long)]
    edge_weight_key: Option<String>,
    #[arg(long, default_value_t = false)]
    no_progress: bool,
}

#[derive(Serialize, Deserialize, Debug)]
struct JsonGraphData {
    directed: bool,
    multigraph: bool,
    graph: Vec<Value>,
    nodes: Vec<Value>,
    adjacency: Vec<Value>,
}

/// Basic struct that represents a graph with nodes and edges.
/// This is a minimal struct compared opted for over something like `petgraph::Graph`
/// in order to reduce overhead since all operations in this file are relatively simple.
#[derive(Debug)]
struct Graph {
    nodes: Vec<Value>,
    edges: HashSet<(u64, u64)>,
    edge_weights: HashMap<(u64, u64), HashMap<String, f64>>,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Args = Args::parse();

    match args.mode {
        Mode::TallyKeys => {
            let graph = make_graph_from_json(match &args.graph_file {
                Some(file) => file,
                _ => panic!("graph file required"),
            })
            .expect("Could not load graph");
            let output_file = &args.ben_file.replace(".jsonl.ben", "_tallies.parquet");

            tally_and_save_from_key_list(
                graph,
                &args.ben_file,
                &output_file,
                args.keys,
                !args.no_progress,
            )?;
        }
        Mode::CutEdges => {
            let graph = make_graph_from_json(match &args.graph_file {
                Some(file) => file,
                _ => panic!("graph file required"),
            })
            .expect("Could not load graph");
            let output_file = &args.ben_file.replace(".jsonl.ben", "_cut_edges.parquet");

            tally_and_save_cut_edges(
                graph,
                &args.ben_file,
                &output_file,
                args.edge_weight_key,
                !args.no_progress,
            )?;
        }
        Mode::ChangedAssignments => {
            tally_and_save_changed_assignments(
                &args.ben_file,
                args.normalize,
                args.max_accepted,
                args.mkv_rand_reassignment_off,
                !args.no_progress,
            )?;
        }
    }
    Ok(())
}

/// Creates a graph from a JSON file.
///
/// # Arguments
///
/// * `file_path` - A string slice that holds the path to the JSON file.
///
/// # Returns
///
/// * `Result<Graph>` - A result containing the graph if successful, or an error if not.
///
/// # Errors
///
/// This function will return an error if the file cannot be found, read, or parsed.
fn make_graph_from_json(file_path: &str) -> Result<Graph> {
    // Read the JSON file
    let mut file = File::open(file_path).expect(format!("File {} not found", file_path).as_str());
    let mut data = String::new();
    file.read_to_string(&mut data).expect("Unable to read file");

    // Parse the JSON data
    let graph_data: JsonGraphData = serde_json::from_str(&data)?;

    let mut graph = Graph {
        nodes: graph_data.nodes.clone(),
        edges: HashSet::new(),
        edge_weights: HashMap::new(),
    };

    for (source_idx, target_array) in graph_data
        .adjacency
        .iter()
        .enumerate()
        .map(|(x, y)| (x as u64, y.as_array().unwrap().to_vec()))
    {
        for target_data in target_array {
            let target_idx: u64 = target_data["id"].as_u64().unwrap();
            graph
                .edges
                .insert((source_idx.min(target_idx), source_idx.max(target_idx)));

            // Pre-parse any numeric attributes present on the edge so keyed weighted cut-edges
            // can be computed without re-walking raw JSON later.
            if let Some(edge_obj) = target_data.as_object() {
                let edge_key = (source_idx.min(target_idx), source_idx.max(target_idx));
                for (key, value) in edge_obj {
                    if key == "id" {
                        continue;
                    }
                    let parsed_weight = match value {
                        Value::Number(n) => n.as_f64(),
                        Value::String(s) => s.parse::<f64>().ok(),
                        _ => None,
                    };
                    if let Some(weight) = parsed_weight {
                        graph
                            .edge_weights
                            .entry((edge_key.0, edge_key.1))
                            .or_insert_with(HashMap::new)
                            .insert(key.clone(), weight);
                    }
                }
            }
        }
    }

    Ok(graph)
}

fn tally_keys(
    graph: &Graph,
    assignment: &Vec<u16>,
    keys: &Vec<String>,
) -> HashMap<String, HashMap<u16, f64>> {
    let partition_values: HashSet<u16> = HashSet::from_iter(assignment.iter().cloned());

    let mut tallies: HashMap<String, HashMap<u16, f64>> = keys
        .iter()
        .map(|x| {
            (
                x.clone(),
                partition_values.iter().map(|&y| (y, 0.0)).collect(),
            )
        })
        .collect();

    for (idx, node) in graph.nodes.iter().enumerate() {
        let partition_key = assignment[idx];
        for key in keys {
            let json_val = &node[key];
            let value = match json_val {
                Value::Number(n) => n.as_f64().unwrap(),
                Value::String(s) => s.parse::<f64>().unwrap(),
                _ => panic!(
                    "Invalid value type in JSON file. Failed to parse {:?} as f64 for key {:?}",
                    json_val, key
                ),
            };

            *tallies
                .get_mut(key)
                .unwrap()
                .get_mut(&partition_key)
                .unwrap() += value;
        }
    }

    tallies
}

/// Given a (parquet) file path to save to and a list of tallies corresponding to values obtained from
/// assessing some function over an ensemble of graph partitions, this function saves the tallies
/// to a Parquet file.
///
/// # Arguments
///
/// * `file_path` - A string slice that holds the path to the Parquet file to save to.
/// * `tallies` - A list of tuples of the following form:
///     - The first element is the sample number.
///     - The second element is the number of repetitions.
///     - The third element is the number of accepted samples.
///     - The fourth element is a hashmap of keys to hashmaps of partition keys to values.
///           the keys of the outer hashmap are the keys to tally, and the inner hashmap
///           keys are the partition numbers (e.g. district numbers) allong with the values
///           for the tally.
///
/// # Returns
///
/// * `Result<(), Box<dyn std::error::Error>>` - A result containing the success or failure of the operation.
fn save_tallies_to_parquet(
    file_path: &str,
    tallies: &Vec<(u64, u32, u32, HashMap<String, HashMap<u16, f64>>)>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut sample_numbers = Vec::new();
    let mut n_reps_numbers = Vec::new();
    let mut accepted_numbers = Vec::new();

    let mut keys = Vec::new();
    let mut partition_data: HashMap<u16, Vec<Option<f64>>> = HashMap::new();

    // Initialize partition_data with empty vectors for each unique partition key
    for (_, _, _, tally) in tallies {
        for (_, sub_map) in tally {
            for (&partition_key, _) in sub_map {
                partition_data.entry(partition_key).or_insert_with(Vec::new);
            }
        }
    }

    // Fill in the data
    for (sample_num, n_reps, accepted_num, tally) in tallies {
        for (key, sub_map) in tally {
            sample_numbers.push(*sample_num);
            n_reps_numbers.push(*n_reps);
            accepted_numbers.push(*accepted_num);
            keys.push(key.clone());
            for (&partition_key, value) in partition_data.iter_mut() {
                value.push(sub_map.get(&partition_key).copied());
            }
        }
    }

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_numbers).into(),
        Series::new("n_reps".into(), n_reps_numbers).into(),
        Series::new("accepted_count".into(), accepted_numbers).into(),
        Series::new("sum_columns".into(), keys).into(),
    ])?;

    // Add columns for each partition key
    for (partition_key, values) in partition_data {
        df.with_column(Series::new(format!("district_{}", partition_key).into(), values).into())?;
    }

    let mut file = File::create(file_path)?;
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(None))
        .finish(&mut df)?;

    Ok(())
}

/// Tallies and saves data from a ensemble of graph partitions to a Parquet file.
///
/// # Arguments
///
/// * `graph` - A `Graph` struct representing the graph that is to be partitioned.
/// * `in_file_name` - A string slice that holds the path to the BEN file to read from.
/// * `out_file_name` - A string slice that holds the path to the Parquet file to save to.
/// * `key_list` - A list of keys to tally.
///
/// # Returns
///
/// * `io::Result<()>` - A result containing the success or failure of the operation.
fn tally_and_save_from_key_list(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    key_list: Vec<String>,
    show_progress: bool,
) -> io::Result<()> {
    let n_pb_tics = 1000;

    let mut pb = if show_progress {
        Some(ProgressBar::new(n_pb_tics))
    } else {
        None
    };

    // pb.set_draw_target(indicatif::ProgressDrawTarget::stderr());
    let mut pb_tics = 0; // For manual printing since my

    let mut ben_file = File::open(in_file_name).expect("BEN file not found");

    let line_checker = BenDecoder::new(&ben_file).expect("Failed to initialize decoder");

    let basename = Path::new(in_file_name)
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy();

    eprintln!("Reading {:?}...", basename);

    let mut line_count: usize = 0;
    for _ in line_checker.enumerate() {
        line_count += 1;
    }
    println!("Found {:?} unique plans in {:?}\r", line_count, basename);

    let pb_step_size = (line_count / n_pb_tics as usize) as u32;
    let mut previous_step = 0;

    ben_file.seek(SeekFrom::Start(0))?;

    let ben_reader = BufReader::new(ben_file);

    let decoder = BenDecoder::new(ben_reader).unwrap();

    let mut all_tallies: Vec<(u64, u32, u32, HashMap<String, HashMap<u16, f64>>)> =
        Vec::with_capacity(line_count);

    let mut sample_count = 1;
    let mut accepted_count = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    let start_time = Instant::now();
    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, n_reps)) => {
                batch.push((assignment, n_reps));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, n_reps)| {
                            let tallies = tally_keys(&graph, assignment, &key_list);
                            (*n_reps, tallies)
                        })
                        .collect();

                    for (n_reps, tallies) in results {
                        all_tallies.push((sample_count, n_reps as u32, accepted_count, tallies));
                        sample_count += n_reps as u64;
                        accepted_count += 1;
                    }

                    batch.clear();
                }
            }
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        }
        if show_progress && accepted_count - previous_step >= pb_step_size {
            let pb_ref = pb.as_mut().unwrap();
            pb_ref.inc();

            let elapsed = start_time.elapsed();
            let elapsed_secs = elapsed.as_secs_f64();
            let rate = (pb_tics + 1) as f64 / elapsed_secs; // Current rate (iterations per second)
            let remaining_secs = (n_pb_tics - pb_tics - 1) as f64 / rate; // Remaining time in seconds

            let elapsed_mins = (elapsed_secs / 60.0).floor() as u64;
            let elapsed_remain_secs = (elapsed_secs % 60.0) as u64;
            let remaining_mins = (remaining_secs / 60.0).floor() as u64;
            let remaining_remain_secs = (remaining_secs % 60.0) as u64;

            // Update the progress bar message to display the formatted elapsed and remaining times
            pb_ref.message(&format!(
                "Elapsed: {}m {}s, ETA: {}m {}s ",
                elapsed_mins, elapsed_remain_secs, remaining_mins, remaining_remain_secs
            ));
            pb_tics += 1;

            io::stderr().flush().unwrap();
            io::stdout().flush().unwrap();
            previous_step = accepted_count;
        }
    }

    // Process any remaining records in the batch
    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|(assignment, n_reps)| {
                let tallies = tally_keys(&graph, assignment, &key_list);
                (*n_reps, tallies)
            })
            .collect();

        for (n_reps, tallies) in results {
            all_tallies.push((sample_count, n_reps as u32, accepted_count, tallies));
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    if let Some(pb_ref) = pb.as_mut() {
        pb_ref.finish();
    }

    eprintln!("Writing final output...");
    save_tallies_to_parquet(out_file_name, &all_tallies).expect("Unable to save tallies");
    eprintln!("Done!");
    Ok(())
}

/// Counts the number of cut edges in the graph based on the assignment
/// of the nodes in the graph.
///
/// # Arguments
///
/// * `graph` - A `Graph` struct representing the graph to count the cut edges in.
/// * `assignment` - A vector of u16 values representing the assignment of the nodes in the graph.
///
/// # Returns
///
/// * `f64` - The weighted number of cut edges in the graph.
fn cut_edges(graph: &Graph, assignment: &Vec<u16>, edge_weight_key: Option<&str>) -> f64 {
    let mut cut_edges = 0.0;

    for edge in &graph.edges {
        let (source, target) = edge;
        if assignment[*source as usize] != assignment[*target as usize] {
            let weight = match edge_weight_key {
                Some(key) => graph
                    .edge_weights
                    .get(&(*source, *target))
                    .and_then(|weights_by_key| weights_by_key.get(key).copied())
                    .unwrap_or(1.0),
                None => 1.0,
            };
            cut_edges += weight;
        }
    }

    cut_edges
}

/// Tallies and saves the number of cut edges in the graph to a Parquet file.
///
/// # Arguments
///
/// * `graph` - A `Graph` struct representing the graph to count the cut edges in.
/// * `in_file_name` - A string slice that holds the path to the BEN file to read from.
/// * `out_file_name` - A string slice that holds the path to the Parquet file to save to.
///
/// # Returns
///
/// * `std::result::Result<(), Box<dyn std::error::Error>>` - A result containing the success or
///     failure of the operation.
fn tally_and_save_cut_edges(
    graph: Graph,
    in_file_name: &str,
    out_file_name: &str,
    edge_weight_key: Option<String>,
    show_progress: bool,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let n_pb_tics = 100;

    let mut pb = if show_progress {
        Some(ProgressBar::new(n_pb_tics))
    } else {
        None
    };

    let ben_file = File::open(in_file_name).expect("BEN file not found");

    let decoder = BenDecoder::new(&ben_file).expect("Failed to initialize decoder");

    let basename = Path::new(in_file_name)
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy();

    eprintln!("Reading {:?}...", basename);

    let line_count = count_samples_from_file(Path::new(in_file_name), "ben")
        .expect("Failed to count samples in BEN file");

    let pb_step_size = (line_count / n_pb_tics as usize) as u32;
    let mut previous_step = 0;

    let mut sample_nums = Vec::with_capacity(line_count);
    let mut n_reps_nums = Vec::with_capacity(line_count);
    let mut accepted_nums = Vec::with_capacity(line_count);
    let mut cut_edge_counts = Vec::with_capacity(line_count);

    let mut sample_count = 1;
    let mut accepted_count = 1;

    const BATCH_SIZE: usize = 100;
    let mut batch = Vec::with_capacity(BATCH_SIZE);

    for (_idx, record) in decoder.enumerate() {
        match record {
            Ok((assignment, count)) => {
                batch.push((assignment, count));
                if batch.len() == BATCH_SIZE {
                    let results: Vec<_> = batch
                        .par_iter()
                        .map(|(assignment, count)| {
                            let cut_edges =
                                cut_edges(&graph, assignment, edge_weight_key.as_deref());
                            (*count, cut_edges)
                        })
                        .collect();

                    for (n_reps, counts) in results {
                        sample_nums.push(sample_count);
                        n_reps_nums.push(n_reps as u32);
                        accepted_nums.push(accepted_count);
                        cut_edge_counts.push(counts);
                        sample_count += n_reps as u64;
                        accepted_count += 1;
                    }

                    batch.clear();
                }
                if show_progress && accepted_count - previous_step >= pb_step_size {
                    pb.as_mut().unwrap().inc();
                    previous_step = accepted_count;
                }
            }
            Err(e) => {
                panic!("Error: {:?}", e);
            }
        }
    }

    if !batch.is_empty() {
        let results: Vec<_> = batch
            .par_iter()
            .map(|(assignment, count)| {
                let cut_edges = cut_edges(&graph, assignment, edge_weight_key.as_deref());
                (*count, cut_edges)
            })
            .collect();

        for (n_reps, counts) in results {
            sample_nums.push(sample_count);
            n_reps_nums.push(n_reps as u32);
            accepted_nums.push(accepted_count);
            cut_edge_counts.push(counts);
            sample_count += n_reps as u64;
            accepted_count += 1;
        }
    }

    if let Some(pb_ref) = pb.as_mut() {
        pb_ref.finish();
    }

    println!();

    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), sample_nums).into(),
        Series::new("n_reps".into(), n_reps_nums).into(),
        Series::new("accepted_count".into(), accepted_nums).into(),
        Series::new("cut_edges".into(), cut_edge_counts).into(),
    ])?;

    let mut file = File::create(out_file_name)?;

    eprintln!("Writing final output...");
    ParquetWriter::new(&mut file)
        .with_compression(ParquetCompression::Brotli(None))
        .finish(&mut df)?;

    eprintln!("Done!");
    Ok(())
}

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
fn tally_and_save_changed_assignments(
    in_ben_file: &str,
    normalize: bool,
    max_accepted: Option<usize>,
    with_random_reassignments: bool,
    show_progress: bool,
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

    let out_file_name = &in_ben_file.replace(
        ".jsonl.ben",
        format!("_accept_{}_changed_assignments.txt", line_count).as_str(),
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

    let mut out = File::create(out_file_name)
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
