use crate::cli::Args;
use crate::commands::{graph_file, require_keys};
use crate::graph::{load_graph, GraphLoadRequest};
use crate::metrics;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    require_keys(&args, "tally-keys")?;
    let graph = load_graph(
        graph_file(&args)?,
        GraphLoadRequest {
            numeric_keys: args.keys.clone(),
            ..Default::default()
        },
    )?;
    let output_dir = args.output_dir.as_deref();
    metrics::tally_keys::tally_and_save_from_key_list(
        graph,
        &args.ben_file,
        output_dir,
        args.keys,
        !args.no_progress,
        args.high_compression,
    )
}
