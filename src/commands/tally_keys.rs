use crate::cli::Args;
use crate::commands::{graph_file, require_keys};
use crate::graph::{load_graph, GraphLoadRequest};
use crate::metrics;

pub fn run(args: Args) -> crate::error::Result<()> {
    require_keys(&args, "tally-keys")?;
    let graph = load_graph(
        graph_file(&args)?,
        GraphLoadRequest {
            numeric_keys: args.keys.clone(),
            ..Default::default()
        },
    )?;
    let show_progress = args.show_progress();
    let output_dir = args.output_dir.as_deref();
    metrics::tally_keys::tally_and_save_from_key_list(
        graph,
        &args.ben_file,
        output_dir,
        args.keys,
        show_progress,
        args.high_compression,
    )
}
