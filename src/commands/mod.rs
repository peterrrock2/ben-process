use crate::cli::{Args, Mode};
use std::io;
use std::path::Path;

mod changed_assignments;
mod cut_edges;
mod extract_unique_plans;
mod polsby_popper;
mod region;
mod tally_keys;
mod unique_plans;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Every mode derives its output name from the BEN file's basename; a path with no file-name
    // component (empty, `.`, `..`, `/`) would otherwise panic in `ben_stem`. Reject it up front.
    if Path::new(&args.ben_file).file_name().is_none() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid --ben-file {:?}: path has no file name component",
                args.ben_file
            ),
        )));
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

pub(super) fn graph_file(args: &Args) -> std::result::Result<&str, Box<dyn std::error::Error>> {
    args.graph_file.as_deref().ok_or_else(|| {
        Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "graph file required; pass --graph-file <PATH>",
        )) as Box<dyn std::error::Error>
    })
}

pub(super) fn require_keys(
    args: &Args,
    mode_name: &str,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    if args.keys.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("at least one key is required for {} mode", mode_name),
        )));
    }

    Ok(())
}
