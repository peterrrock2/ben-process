use crate::cli::{build_output_path, Args};
use crate::commands::{graph_file, require_keys};
use crate::graph::{load_graph, GraphLoadRequest};
use crate::metrics;
use crate::metrics::region::RegionMetric;

pub fn run_splits(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run(
        args,
        RegionMetric::Splits,
        "_region_splits.parquet",
        "region-splits",
    )
}

pub fn run_pieces(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    run(
        args,
        RegionMetric::Pieces,
        "_region_pieces.parquet",
        "region-pieces",
    )
}

fn run(
    args: Args,
    metric: RegionMetric,
    suffix: &str,
    mode_name: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    require_keys(&args, mode_name)?;
    let graph = load_graph(
        graph_file(&args)?,
        GraphLoadRequest {
            region_keys: args.keys.clone(),
            ..Default::default()
        },
    )?;
    let output_file = build_output_path(&args.ben_file, suffix, args.output_dir.as_deref());

    metrics::region::tally_and_save_region_metric(
        graph,
        &args.ben_file,
        output_file.as_str(),
        args.keys,
        metric,
        !args.no_progress,
        args.high_compression,
    )
}
