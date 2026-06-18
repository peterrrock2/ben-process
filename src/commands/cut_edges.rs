use crate::cli::{build_output_path, Args};
use crate::commands::resolve_graph;
use crate::graph::{EdgeWeightRequest, GraphLoadRequest};
use crate::input;
use crate::metrics;

pub fn run(args: Args) -> crate::error::Result<()> {
    let resolved = input::resolve(&args.ben_file)?;
    let graph = resolve_graph(
        &args,
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
        &args.ben_file,
        "_cut_edges.parquet",
        args.output_dir.as_deref(),
    );

    metrics::cut_edges::tally_and_save_cut_edges(
        graph,
        &resolved.source,
        output_file.as_str(),
        args.show_progress(),
        args.high_compression,
    )
}
