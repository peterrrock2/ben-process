use crate::cli::{build_output_path, Args};
use crate::commands::graph_file;
use crate::graph::{load_graph, EdgeWeightRequest, GraphLoadRequest};
use crate::metrics;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let area_key = args.area_key.clone().unwrap_or_else(|| "area".to_string());
    let shared_perim_key = args
        .shared_perim_key
        .clone()
        .unwrap_or_else(|| "shared_perim".to_string());
    // `--perim-key` (direct node perimeter) takes precedence; only when it is absent do we fall
    // back to deriving from boundary perimeter, defaulting that key to the GerryChain
    // "boundary_perim".
    let (perim_key, boundary_perim_key) = match &args.perim_key {
        Some(perim_key) => (Some(perim_key.clone()), args.boundary_perim_key.clone()),
        None => (
            None,
            Some(
                args.boundary_perim_key
                    .clone()
                    .unwrap_or_else(|| "boundary_perim".to_string()),
            ),
        ),
    };

    let mut numeric_keys = vec![area_key.clone()];
    let mut partial_numeric_keys = Vec::new();
    if let Some(perim_key) = &perim_key {
        numeric_keys.push(perim_key.clone());
    } else if let Some(boundary_perim_key) = &boundary_perim_key {
        partial_numeric_keys.push(boundary_perim_key.clone());
    }

    let graph = load_graph(
        graph_file(&args)?,
        GraphLoadRequest {
            numeric_keys,
            partial_numeric_keys,
            edge_weight: Some(EdgeWeightRequest {
                key: shared_perim_key.clone(),
                default_value: 0.0,
            }),
            ..Default::default()
        },
    )?;
    let output_file = build_output_path(
        &args.ben_file,
        "_polsby_popper.parquet",
        args.output_dir.as_deref(),
    );

    metrics::polsby_popper::tally_and_save_polsby_popper(
        graph,
        &args.ben_file,
        output_file.as_str(),
        area_key.as_str(),
        perim_key.as_deref(),
        boundary_perim_key.as_deref(),
        !args.no_progress,
        args.high_compression,
    )
}
