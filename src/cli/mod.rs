use clap::{Parser, Subcommand};

use crate::graph::{load_graph, load_graph_from_reader, Graph, GraphLoadRequest};
use crate::input::ResolvedInput;
use std::io::{self, Cursor};
use std::path::Path;

mod changed_assignments;
mod cut_edges;
mod extract_unique_plans;
mod paths;
mod polsby_popper;
mod region;
mod reock;
mod tally_keys;
mod unique_plans;

pub use changed_assignments::ChangedAssignmentsArgs;
pub use cut_edges::CutEdgesArgs;
pub use extract_unique_plans::ExtractUniquePlansArgs;
pub use paths::*;
pub use polsby_popper::PolsbyPopperArgs;
pub use region::RegionArgs;
pub use reock::ReockArgs;
pub use tally_keys::TallyKeysArgs;
pub use unique_plans::UniquePlansArgs;

#[derive(Parser, Debug)]
#[command(
    name = "ben-process",
    about = "A tool for processing BEN files and saving outputs to Parquet or text.",
    version
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
    /// Enable info-level status logging. `RUST_LOG` overrides this.
    #[arg(short, long, global = true)]
    pub verbose: bool,
    /// Suppress the progress bar. Errors and warnings are still printed.
    #[arg(short, long, global = true)]
    pub quiet: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Tally numeric node attributes by district.
    TallyKeys(TallyKeysArgs),
    /// Count cut edges per plan (optionally edge-weighted).
    CutEdges(CutEdgesArgs),
    /// District-level Polsby-Popper scores per plan.
    PolsbyPopper(PolsbyPopperArgs),
    /// District-level Reock scores per plan.
    Reock(ReockArgs),
    /// Per-node assignment change counts across accepted plans.
    ChangedAssignments(ChangedAssignmentsArgs),
    /// Count split regions per plan.
    RegionSplits(RegionArgs),
    /// Total district pieces per region per plan.
    RegionPieces(RegionArgs),
    /// Count label-invariant unique partitions.
    UniquePlans(UniquePlansArgs),
    /// Extract the first occurrence of each unique partition as a Standard BEN.
    ExtractUniquePlans(ExtractUniquePlansArgs),
}

impl Command {
    /// Shared input/output flags, for the central file-name guard in `run`.
    fn common(&self) -> &CommonArgs {
        match self {
            Command::TallyKeys(a) => &a.common,
            Command::CutEdges(a) => &a.common,
            Command::PolsbyPopper(a) => &a.common,
            Command::Reock(a) => &a.common,
            Command::ChangedAssignments(a) => &a.common,
            Command::RegionSplits(a) => &a.common,
            Command::RegionPieces(a) => &a.common,
            Command::UniquePlans(a) => &a.common,
            Command::ExtractUniquePlans(a) => &a.common,
        }
    }
}

/// Input/output flags shared by every subcommand.
#[derive(clap::Args, Debug)]
pub struct CommonArgs {
    #[arg(short = 'b', long = "ben-file", value_name = "BEN_FILE")]
    pub ben_file: String,
    #[arg(long)]
    pub output_dir: Option<String>,
}

impl CommonArgs {
    pub fn ben_file(&self) -> &str {
        &self.ben_file
    }

    pub fn output_dir(&self) -> Option<&str> {
        self.output_dir.as_deref()
    }
}

pub fn run(cli: Cli) -> crate::error::Result<()> {
    let show_progress = !cli.quiet;

    // Every mode derives its output name from the BEN file's basename; a path with no file-name
    // component (empty, `.`, `..`, `/`) would otherwise panic in `ben_stem`. Reject it up front.
    let ben_file = cli.command.common().ben_file();
    if Path::new(ben_file).file_name().is_none() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid --ben-file {:?}: path has no file name component",
                ben_file
            ),
        )
        .into());
    }

    match cli.command {
        Command::TallyKeys(args) => tally_keys::run(args, show_progress),
        Command::CutEdges(args) => cut_edges::run(args, show_progress),
        Command::PolsbyPopper(args) => polsby_popper::run(args, show_progress),
        Command::Reock(args) => reock::run(args, show_progress),
        Command::ChangedAssignments(args) => changed_assignments::run(args, show_progress),
        Command::RegionSplits(args) => region::run_splits(args, show_progress),
        Command::RegionPieces(args) => region::run_pieces(args, show_progress),
        Command::UniquePlans(args) => unique_plans::run(args, show_progress),
        Command::ExtractUniquePlans(args) => extract_unique_plans::run(args, show_progress),
    }
}

/// Pick the dual graph for a graph-driven mode. Precedence: an explicit `--graph-file` always wins;
/// otherwise a graph embedded in a `.bendl` bundle is used; otherwise the standard "graph file
/// required" error. A `--graph-file` that overrides a bundle's embedded graph is logged.
pub(super) fn resolve_graph(
    graph_file: Option<&str>,
    resolved: &ResolvedInput,
    request: GraphLoadRequest,
) -> crate::error::Result<Graph> {
    match (graph_file, &resolved.embedded_graph) {
        (Some(path), embedded) => {
            if embedded.is_some() {
                log::info!("--graph-file given; ignoring the graph embedded in the bundle");
            }
            load_graph(path, request)
        }
        (None, Some(bytes)) => load_graph_from_reader(Cursor::new(bytes), request),
        (None, None) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "graph file required; pass --graph-file <PATH>",
        )
        .into()),
    }
}

pub(super) fn require_keys(keys: &[String], mode_name: &str) -> crate::error::Result<()> {
    if keys.is_empty() {
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
    for key in keys {
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
