//! Entry point for `ben-tally`. Dispatches CLI mode → per-mode runner.
//!
//! The bulk of the logic lives in the submodules:
//!
//! - [`cli`]: clap args + output path helper
//! - [`graph`]: graph struct + JSON loader
//! - [`metrics`]: per-sample compute (tally_keys, cut_edges, region)
//! - [`changed_assignments`]: sequential across-sample mode

use clap::Parser;

mod changed_assignments;
mod cli;
mod graph;
mod metrics;

use cli::{build_output_path, Args, Mode};
use graph::make_graph_from_json;
use metrics::region::RegionMetric;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Args = Args::parse();

    match args.mode {
        Mode::TallyKeys => {
            let graph = make_graph_from_json(match &args.graph_file {
                Some(file) => file,
                _ => panic!("graph file required"),
            })
            .expect("Could not load graph");
            let output_file = build_output_path(
                &args.ben_file,
                "_tallies.parquet",
                args.output_dir.as_deref(),
            );

            metrics::tally_keys::tally_and_save_from_key_list(
                graph,
                &args.ben_file,
                output_file.as_str(),
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
            let output_file = build_output_path(
                &args.ben_file,
                "_cut_edges.parquet",
                args.output_dir.as_deref(),
            );

            metrics::cut_edges::tally_and_save_cut_edges(
                graph,
                &args.ben_file,
                output_file.as_str(),
                args.edge_weight_key,
                !args.no_progress,
            )?;
        }
        Mode::ChangedAssignments => {
            changed_assignments::tally_and_save_changed_assignments(
                &args.ben_file,
                args.normalize,
                args.max_accepted,
                args.mkv_rand_reassignment_off,
                !args.no_progress,
                args.output_dir.as_deref(),
            )?;
        }
        Mode::RegionSplits => {
            let graph = make_graph_from_json(match &args.graph_file {
                Some(file) => file,
                _ => panic!("graph file required"),
            })
            .expect("Could not load graph");
            let output_file = build_output_path(
                &args.ben_file,
                "_region_splits.parquet",
                args.output_dir.as_deref(),
            );
            if args.keys.is_empty() {
                panic!("at least one key is required for region-splits mode");
            }

            metrics::region::tally_and_save_region_metric(
                graph,
                &args.ben_file,
                output_file.as_str(),
                args.keys,
                RegionMetric::Splits,
                !args.no_progress,
            )?;
        }
        Mode::RegionPieces => {
            let graph = make_graph_from_json(match &args.graph_file {
                Some(file) => file,
                _ => panic!("graph file required"),
            })
            .expect("Could not load graph");
            let output_file = build_output_path(
                &args.ben_file,
                "_region_pieces.parquet",
                args.output_dir.as_deref(),
            );
            if args.keys.is_empty() {
                panic!("at least one key is required for region-pieces mode");
            }

            metrics::region::tally_and_save_region_metric(
                graph,
                &args.ben_file,
                output_file.as_str(),
                args.keys,
                RegionMetric::Pieces,
                !args.no_progress,
            )?;
        }
    }
    Ok(())
}
