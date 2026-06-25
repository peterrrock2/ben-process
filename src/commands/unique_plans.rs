use crate::cli::{build_output_path, Args};
use crate::input;
use crate::metrics;

pub fn run(args: Args) -> crate::error::Result<()> {
    let resolved = input::resolve(&args.ben_file)?;
    let output_file = build_output_path(
        &args.ben_file,
        "_unique_plans.parquet",
        args.output_dir.as_deref(),
    );

    metrics::unique_plans::count_and_save_unique_plans(
        &resolved.source,
        output_file.as_str(),
        args.show_progress(),
        args.max_samples,
        args.high_compression,
    )
}
