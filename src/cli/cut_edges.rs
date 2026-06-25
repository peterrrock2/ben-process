use super::{build_output_path, resolve_graph, CommonArgs};
use crate::graph::{EdgeWeightRequest, GraphLoadRequest};
use crate::input;
use crate::metrics;

#[derive(clap::Args, Debug)]
pub struct CutEdgesArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(short, long)]
    pub graph_file: Option<String>,
    #[arg(long)]
    pub edge_weight_key: Option<String>,
    /// Stop after this many expanded samples.
    #[arg(long)]
    pub max_samples: Option<usize>,
    /// Use Brotli compression for Parquet output (default: Snappy).
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
}

pub fn run(args: CutEdgesArgs, show_progress: bool) -> crate::error::Result<()> {
    let resolved = input::resolve(args.common.ben_file())?;
    let graph = resolve_graph(
        args.graph_file.as_deref(),
        &resolved,
        GraphLoadRequest {
            edge_weight: args.edge_weight_key.clone().map(|key| EdgeWeightRequest {
                key,
                default_value: 1.0,
            }),
            ..Default::default()
        },
    )?;
    let output_file = build_output_path(
        args.common.ben_file(),
        "_cut_edges.parquet",
        args.common.output_dir(),
    );

    metrics::cut_edges::tally_and_save_cut_edges(
        graph,
        &resolved.source,
        output_file.as_str(),
        show_progress,
        args.max_samples,
        args.high_compression,
    )
}
