use crate::cli::Args;

pub fn run(args: Args) -> crate::error::Result<()> {
    crate::metrics::changed_assignments::tally_and_save_changed_assignments(
        &args.ben_file,
        args.normalize,
        args.max_accepted,
        args.randomize_reassignments,
        args.seed,
        args.show_progress(),
        args.output_dir.as_deref(),
        args.high_compression,
    )
}
