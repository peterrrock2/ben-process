use crate::cli::{Args, Mode};
use crate::graph::{load_graph, load_graph_from_reader, Graph, GraphLoadRequest};
use crate::input::ResolvedInput;
use std::io::{self, Cursor};
use std::path::Path;

mod changed_assignments;
mod cut_edges;
mod extract_unique_plans;
mod polsby_popper;
mod region;
mod tally_keys;
mod unique_plans;

pub fn run(args: Args) -> crate::error::Result<()> {
    // Every mode derives its output name from the BEN file's basename; a path with no file-name
    // component (empty, `.`, `..`, `/`) would otherwise panic in `ben_stem`. Reject it up front.
    if Path::new(&args.ben_file).file_name().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid --ben-file {:?}: path has no file name component",
                args.ben_file
            ),
        )
        .into());
    }

    match args.mode.clone() {
        Mode::TallyKeys => tally_keys::run(args),
        Mode::CutEdges => cut_edges::run(args),
        Mode::PolsbyPopper => polsby_popper::run(args),
        Mode::ChangedAssignments => changed_assignments::run(args),
        Mode::RegionSplits => region::run_splits(args),
        Mode::RegionPieces => region::run_pieces(args),
        Mode::UniquePlans => unique_plans::run(args),
        Mode::ExtractUniquePlans => extract_unique_plans::run(args),
    }
}

/// Pick the dual graph for a graph-driven mode. Precedence: an explicit `--graph-file` always wins;
/// otherwise a graph embedded in a `.bendl` bundle is used; otherwise the standard "graph file
/// required" error. A `--graph-file` that overrides a bundle's embedded graph is logged.
pub(super) fn resolve_graph(
    args: &Args,
    resolved: &ResolvedInput,
    request: GraphLoadRequest,
) -> crate::error::Result<Graph> {
    match (&args.graph_file, &resolved.embedded_graph) {
        (Some(path), embedded) => {
            if embedded.is_some() {
                log::info!("--graph-file given; ignoring the graph embedded in the bundle");
            }
            load_graph(path, request)
        }
        (None, Some(bytes)) => load_graph_from_reader(Cursor::new(bytes), request),
        (None, None) => load_graph(graph_file(args)?, request),
    }
}

pub(super) fn graph_file(args: &Args) -> crate::error::Result<&str> {
    args.graph_file.as_deref().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "graph file required; pass --graph-file <PATH>",
        )
        .into()
    })
}

pub(super) fn require_keys(args: &Args, mode_name: &str) -> crate::error::Result<()> {
    if args.keys.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("at least one key is required for {} mode", mode_name),
        )
        .into());
    }

    // Duplicate keys are never intentional, and for tally-keys they are actively harmful: each key
    // derives an output path, so two writers would open the same file and interleave row groups
    // into unreadable Parquet.
    let mut seen = std::collections::HashSet::new();
    for key in &args.keys {
        if !seen.insert(key.as_str()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate key {:?} passed to --keys", key),
            )
            .into());
        }
    }

    Ok(())
}
