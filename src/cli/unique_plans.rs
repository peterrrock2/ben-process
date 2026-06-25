use super::{build_output_path, CommonArgs};
use crate::input;
use crate::metrics;

#[derive(clap::Args, Debug)]
pub struct UniquePlansArgs {
    #[command(flatten)]
    pub common: CommonArgs,
    /// Stop after this many expanded samples.
    #[arg(long)]
    pub max_samples: Option<usize>,
    /// Use Brotli compression for Parquet output (default: Snappy).
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
}

pub fn run(args: UniquePlansArgs, show_progress: bool) -> crate::error::Result<()> {
    let resolved = input::resolve(args.common.ben_file())?;
    let output_file = build_output_path(
        args.common.ben_file(),
        "_unique_plans.parquet",
        args.common.output_dir(),
    );

    metrics::unique_plans::count_and_save_unique_plans(
        &resolved.source,
        output_file.as_str(),
        show_progress,
        args.max_samples,
        args.high_compression,
    )
}
