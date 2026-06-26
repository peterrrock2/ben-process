use super::{build_output_path, resolve_graph, CommonArgs};
use crate::graph::{EdgeWeightRequest, GraphLoadRequest};
use crate::input;
use crate::metrics;
use ben::BenVariant;

#[derive(clap::Args, Debug)]
pub struct PolsbyPopperArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(short, long)]
    pub graph_file: Option<String>,
    #[arg(long)]
    pub area_key: Option<String>,
    #[arg(long)]
    pub perim_key: Option<String>,
    #[arg(long)]
    pub boundary_perim_key: Option<String>,
    #[arg(long)]
    pub shared_perim_key: Option<String>,
    /// Stop after this many expanded samples.
    #[arg(long)]
    pub max_samples: Option<usize>,
    /// Use Brotli compression for Parquet output (default: Snappy).
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
}

pub fn run(args: PolsbyPopperArgs, show_progress: bool) -> crate::error::Result<()> {
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

    let resolved = input::resolve(args.common.ben_file())?;
    let need_adjacency = resolved.source.variant()? == BenVariant::TwoDelta;
    let graph = resolve_graph(
        args.graph_file.as_deref(),
        &resolved,
        GraphLoadRequest {
            numeric_keys,
            partial_numeric_keys,
            edge_weight: Some(EdgeWeightRequest {
                key: shared_perim_key.clone(),
                default_value: 0.0,
            }),
            need_adjacency,
            ..Default::default()
        },
    )?;
    let output_file = build_output_path(
        args.common.ben_file(),
        "_polsby_popper.parquet",
        args.common.output_dir(),
    );

    metrics::polsby_popper::tally_and_save_polsby_popper(
        graph,
        &resolved.source,
        output_file.as_str(),
        area_key.as_str(),
        perim_key.as_deref(),
        boundary_perim_key.as_deref(),
        show_progress,
        args.max_samples,
        args.high_compression,
    )
}
