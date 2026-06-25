use super::{require_keys, resolve_graph, CommonArgs};
use crate::graph::GraphLoadRequest;
use crate::input;
use crate::metrics;

#[derive(clap::Args, Debug)]
pub struct TallyKeysArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(short, long)]
    pub graph_file: Option<String>,
    #[arg(short, long, num_args(1..))]
    pub keys: Vec<String>,
    /// Stop after this many expanded samples.
    #[arg(long)]
    pub max_samples: Option<usize>,
    /// Use Brotli compression for Parquet output (default: Snappy). Brotli is CPU-heavy and rarely
    /// worth it unless you're storage-bound.
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
}

pub fn run(args: TallyKeysArgs, show_progress: bool) -> crate::error::Result<()> {
    require_keys(&args.keys, "tally-keys")?;
    let resolved = input::resolve(args.common.ben_file())?;
    let graph = resolve_graph(
        args.graph_file.as_deref(),
        &resolved,
        GraphLoadRequest {
            numeric_keys: args.keys.clone(),
            ..Default::default()
        },
    )?;
    let output_dir = args.common.output_dir();
    metrics::tally_keys::tally_and_save_from_key_list(
        graph,
        &resolved.source,
        output_dir,
        args.keys,
        show_progress,
        args.max_samples,
        args.high_compression,
    )
}
