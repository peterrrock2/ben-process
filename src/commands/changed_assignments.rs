use crate::cli::Args;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    crate::metrics::changed_assignments::tally_and_save_changed_assignments(
        &args.ben_file,
        args.normalize,
        args.max_accepted,
        args.randomize_reassignments,
        args.seed,
        args.show_progress(),
        args.output_dir.as_deref(),
    )
}
