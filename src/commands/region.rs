use crate::cli::{build_output_path, Args};
use crate::commands::{require_keys, resolve_graph};
use crate::graph::GraphLoadRequest;
use crate::input;
use crate::metrics;
use crate::metrics::region::RegionMetric;

pub fn run_splits(args: Args) -> crate::error::Result<()> {
    run(
        args,
        RegionMetric::Splits,
        "_region_splits.parquet",
        "region-splits",
    )
}

pub fn run_pieces(args: Args) -> crate::error::Result<()> {
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
) -> crate::error::Result<()> {
    require_keys(&args, mode_name)?;
    let resolved = input::resolve(&args.ben_file)?;
    let graph = resolve_graph(
        &args,
        &resolved,
        GraphLoadRequest {
            region_keys: args.keys.clone(),
            ..Default::default()
        },
    )?;
    let output_file = build_output_path(&args.ben_file, suffix, args.output_dir.as_deref());
    let show_progress = args.show_progress();

    metrics::region::tally_and_save_region_metric(
        graph,
        &resolved.source,
        output_file.as_str(),
        args.keys,
        metric,
        show_progress,
        args.max_samples,
        args.high_compression,
    )
}
