use super::CommonArgs;
use crate::input;

#[derive(clap::Args, Debug)]
pub struct ChangedAssignmentsArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    #[arg(short, long, default_value_t = false)]
    pub normalize: bool,
    #[arg(long)]
    pub max_accepted: Option<usize>,
    /// Randomize merge-split label reassignments. Only set this for MCMC merge-split ensembles.
    /// Default: off.
    #[arg(long, default_value_t = false)]
    pub randomize_reassignments: bool,
    /// Seed for `--randomize-reassignments`. When omitted, a fresh OS-seeded RNG is used and the
    /// randomized run is not reproducible.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Use Brotli compression for Parquet output (default: Snappy).
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
}

pub fn run(args: ChangedAssignmentsArgs, show_progress: bool) -> crate::error::Result<()> {
    let resolved = input::resolve(args.common.ben_file())?;
    crate::metrics::changed_assignments::tally_and_save_changed_assignments(
        &resolved.source,
        args.normalize,
        args.max_accepted,
        args.randomize_reassignments,
        args.seed,
        show_progress,
        args.common.output_dir(),
        args.high_compression,
    )
}
