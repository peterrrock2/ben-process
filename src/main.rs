//! Entry point for `ben-tally`. Dispatches CLI mode → per-mode runner.
//!
//! Each mode asks `load_graph` to pre-parse only the columns it needs:
//! tally-keys wants numeric node attrs, region-* want interned region ids,
//! cut-edges wants an edge-weight vector (or none).

use clap::Parser;

mod changed_assignments;
mod cli;
mod graph;
mod metrics;
mod pipeline;

use cli::{build_output_path, Args, Mode};
use graph::load_graph;
use metrics::region::RegionMetric;

fn graph_file_or_die(args: &Args) -> &str {
    args.graph_file
        .as_deref()
        .unwrap_or_else(|| panic!("graph file required"))
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Args = Args::parse();

    match args.mode {
        Mode::TallyKeys => {
            let graph = load_graph(graph_file_or_die(&args), &args.keys, &[], None)
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
                args.high_compression,
            )?;
        }
        Mode::CutEdges => {
            let graph = load_graph(
                graph_file_or_die(&args),
                &[],
                &[],
                args.edge_weight_key.as_deref(),
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
            let graph = load_graph(graph_file_or_die(&args), &[], &args.keys, None)
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
            let graph = load_graph(graph_file_or_die(&args), &[], &args.keys, None)
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
    }
    Ok(())
}
