use crate::cli::Args;
use crate::input;

pub fn run(args: Args) -> crate::error::Result<()> {
    let resolved = input::resolve(&args.ben_file)?;
    crate::metrics::changed_assignments::tally_and_save_changed_assignments(
        &resolved.source,
        args.normalize,
        args.max_accepted,
        args.randomize_reassignments,
        args.seed,
        args.show_progress(),
        args.output_dir.as_deref(),
        args.high_compression,
    )
}
