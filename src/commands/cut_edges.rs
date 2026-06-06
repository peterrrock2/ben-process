use crate::cli::{build_output_path, Args};
use crate::commands::graph_file_or_die;
use crate::graph::{load_graph, EdgeWeightRequest, GraphLoadRequest};
use crate::metrics;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let graph = load_graph(
        graph_file_or_die(&args),
        GraphLoadRequest {
            edge_weight: args.edge_weight_key.clone().map(|key| EdgeWeightRequest {
                key,
                default_value: 1.0,
            }),
            ..Default::default()
        },
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
    )
}
