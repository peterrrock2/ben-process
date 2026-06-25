use super::{build_output_path, CommonArgs};
use crate::input;
use crate::metrics;

#[derive(clap::Args, Debug)]
pub struct ExtractUniquePlansArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}

pub fn run(args: ExtractUniquePlansArgs, show_progress: bool) -> crate::error::Result<()> {
    let resolved = input::resolve(args.common.ben_file())?;
    let output_file = build_output_path(
        args.common.ben_file(),
        "_unique.jsonl.ben",
        args.common.output_dir(),
    );

    metrics::extract_unique_plans::extract_unique_plans(
        &resolved.source,
        output_file.as_str(),
        show_progress,
    )
}
