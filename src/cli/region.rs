use super::{build_output_path, require_keys, resolve_graph, CommonArgs};
use crate::graph::GraphLoadRequest;
use crate::input;
use crate::metrics;
use crate::metrics::region::RegionMetric;

#[derive(clap::Args, Debug)]
pub struct RegionArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(short, long)]
    pub graph_file: Option<String>,
    #[arg(short, long, num_args(1..))]
    pub keys: Vec<String>,
    /// Stop after this many expanded samples.
    #[arg(long)]
    pub max_samples: Option<usize>,
    /// Use Brotli compression for Parquet output (default: Snappy).
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
}

pub fn run_splits(args: RegionArgs, show_progress: bool) -> crate::error::Result<()> {
    run(
        args,
        show_progress,
        RegionMetric::Splits,
        "_region_splits.parquet",
        "region-splits",
    )
}

pub fn run_pieces(args: RegionArgs, show_progress: bool) -> crate::error::Result<()> {
    run(
        args,
        show_progress,
        RegionMetric::Pieces,
        "_region_pieces.parquet",
        "region-pieces",
    )
}

fn run(
    args: RegionArgs,
    show_progress: bool,
    metric: RegionMetric,
    suffix: &str,
    mode_name: &str,
) -> crate::error::Result<()> {
    require_keys(&args.keys, mode_name)?;
    let resolved = input::resolve(args.common.ben_file())?;
    let graph = resolve_graph(
        args.graph_file.as_deref(),
        &resolved,
        GraphLoadRequest {
            region_keys: args.keys.clone(),
            ..Default::default()
        },
    )?;
    let output_file = build_output_path(args.common.ben_file(), suffix, args.common.output_dir());

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
