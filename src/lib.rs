//! Library entry point for `ben-process`.
//!
//! The binary parses CLI args and delegates here; mode setup and dispatch live
//! behind this seam so they can be tested and refactored without growing
//! `main.rs`.

pub mod changed_assignments;
pub mod cli;
pub mod graph;
pub mod metrics;
pub mod pipeline;

use cli::{build_output_path, Args, Mode};
use graph::load_graph;
use metrics::region::RegionMetric;

fn graph_file_or_die(args: &Args) -> &str {
    args.graph_file
        .as_deref()
        .unwrap_or_else(|| panic!("graph file required"))
}

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    match args.mode {
        Mode::TallyKeys => {
            let graph = load_graph(graph_file_or_die(&args), &args.keys, &[], &[], None, 0.0)
                .expect("Could not load graph");
            let output_dir = args.output_dir.as_deref();
            metrics::tally_keys::tally_and_save_from_key_list(
                graph,
                &args.ben_file,
                output_dir,
                args.keys,
                !args.no_progress,
                args.high_compression,
            )?;
        }
        Mode::CutEdges => {
            let graph = load_graph(
                graph_file_or_die(&args),
                &[],
                &[],
                &[],
                args.edge_weight_key.as_deref(),
                1.0,
            )
            .expect("Could not load graph");
            let output_file = build_output_path(
                &args.ben_file,
                "_cut_edges.parquet",
                args.output_dir.as_deref(),
            );

            metrics::cut_edges::tally_and_save_cut_edges(
                graph,
                &args.ben_file,
                output_file.as_str(),
                !args.no_progress,
                args.high_compression,
            )?;
        }
        Mode::PolsbyPopper => {
            let area_key = args.area_key.clone().unwrap_or_else(|| "area".to_string());
            let shared_perim_key = args
                .shared_perim_key
                .clone()
                .unwrap_or_else(|| "shared_perim".to_string());
            let (perim_key, boundary_perim_key) = match (&args.perim_key, &args.boundary_perim_key)
            {
                (Some(perim_key), _) => Some(perim_key.clone()),
                (None, _) => None,
            }
            .map_or_else(
                || {
                    (
                        None,
                        Some(
                            args.boundary_perim_key
                                .clone()
                                .unwrap_or_else(|| "boundary_perim".to_string()),
                        ),
                    )
                },
                |perim_key| (Some(perim_key), args.boundary_perim_key.clone()),
            );

            let mut numeric_keys = vec![area_key.clone()];
            let mut partial_numeric_keys = Vec::new();
            if let Some(perim_key) = &perim_key {
                numeric_keys.push(perim_key.clone());
            } else if let Some(boundary_perim_key) = &boundary_perim_key {
                partial_numeric_keys.push(boundary_perim_key.clone());
            }

            let graph = load_graph(
                graph_file_or_die(&args),
                &numeric_keys,
                &partial_numeric_keys,
                &[],
                Some(shared_perim_key.as_str()),
                0.0,
            )
            .expect("Could not load graph");
            let output_file = build_output_path(
                &args.ben_file,
                "_polsby_popper.parquet",
                args.output_dir.as_deref(),
            );

            metrics::polsby_popper::tally_and_save_polsby_popper(
                graph,
                &args.ben_file,
                output_file.as_str(),
                area_key.as_str(),
                perim_key.as_deref(),
                boundary_perim_key.as_deref(),
                shared_perim_key.as_str(),
                !args.no_progress,
                args.high_compression,
            )?;
        }
        Mode::ChangedAssignments => {
            changed_assignments::tally_and_save_changed_assignments(
                &args.ben_file,
                args.normalize,
                args.max_accepted,
                args.randomize_reassignments,
                !args.no_progress,
                args.output_dir.as_deref(),
            )?;
        }
        Mode::RegionSplits => {
            if args.keys.is_empty() {
                panic!("at least one key is required for region-splits mode");
            }
            let graph = load_graph(graph_file_or_die(&args), &[], &[], &args.keys, None, 0.0)
                .expect("Could not load graph");
            let output_file = build_output_path(
                &args.ben_file,
                "_region_splits.parquet",
                args.output_dir.as_deref(),
            );

            metrics::region::tally_and_save_region_metric(
                graph,
                &args.ben_file,
                output_file.as_str(),
                args.keys,
                RegionMetric::Splits,
                !args.no_progress,
                args.high_compression,
            )?;
        }
        Mode::RegionPieces => {
            if args.keys.is_empty() {
                panic!("at least one key is required for region-pieces mode");
            }
            let graph = load_graph(graph_file_or_die(&args), &[], &[], &args.keys, None, 0.0)
                .expect("Could not load graph");
            let output_file = build_output_path(
                &args.ben_file,
                "_region_pieces.parquet",
                args.output_dir.as_deref(),
            );

            metrics::region::tally_and_save_region_metric(
                graph,
                &args.ben_file,
                output_file.as_str(),
                args.keys,
                RegionMetric::Pieces,
                !args.no_progress,
                args.high_compression,
            )?;
        }
        Mode::UniquePlans => {
            let output_file = build_output_path(
                &args.ben_file,
                "_unique_plans.txt",
                args.output_dir.as_deref(),
            );

            metrics::unique_plans::count_and_save_unique_plans(
                &args.ben_file,
                output_file.as_str(),
                !args.no_progress,
            )?;
        }
        Mode::ExtractUniquePlans => {
            let output_file = build_output_path(
                &args.ben_file,
                "_unique.jsonl.ben",
                args.output_dir.as_deref(),
            );

            metrics::extract_unique_plans::extract_unique_plans(
                &args.ben_file,
                output_file.as_str(),
                !args.no_progress,
            )?;
        }
    }
    Ok(())
}
