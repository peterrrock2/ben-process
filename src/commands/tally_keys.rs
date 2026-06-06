use crate::cli::Args;
use crate::commands::graph_file_or_die;
use crate::graph::{load_graph, GraphLoadRequest};
use crate::metrics;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let graph = load_graph(
        graph_file_or_die(&args),
        GraphLoadRequest {
            numeric_keys: args.keys.clone(),
            ..Default::default()
        },
    )
    .expect("Could not load graph");
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
