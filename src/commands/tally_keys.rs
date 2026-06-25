use crate::cli::Args;
use crate::commands::{require_keys, resolve_graph};
use crate::graph::GraphLoadRequest;
use crate::input;
use crate::metrics;

pub fn run(args: Args) -> crate::error::Result<()> {
    require_keys(&args, "tally-keys")?;
    let resolved = input::resolve(&args.ben_file)?;
    let graph = resolve_graph(
        &args,
        &resolved,
        GraphLoadRequest {
            numeric_keys: args.keys.clone(),
            ..Default::default()
        },
    )?;
    let show_progress = args.show_progress();
    let output_dir = args.output_dir.as_deref();
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
