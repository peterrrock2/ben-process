use crate::cli::{Args, Mode};

mod changed_assignments;
mod cut_edges;
mod extract_unique_plans;
mod polsby_popper;
mod region;
mod tally_keys;
mod unique_plans;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
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

pub(super) fn graph_file_or_die(args: &Args) -> &str {
    args.graph_file
        .as_deref()
        .unwrap_or_else(|| panic!("graph file required"))
}
