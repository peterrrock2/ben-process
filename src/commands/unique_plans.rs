use crate::cli::{build_output_path, Args};
use crate::metrics;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let output_file = build_output_path(
        &args.ben_file,
        "_unique_plans.txt",
        args.output_dir.as_deref(),
    );

    metrics::unique_plans::count_and_save_unique_plans(
        &args.ben_file,
        output_file.as_str(),
        !args.no_progress,
    )
}
