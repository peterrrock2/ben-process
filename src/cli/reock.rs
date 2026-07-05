use super::{build_output_path, CommonArgs};
use crate::geometry::{load_reock_units_from_geoparquet, ReockLoadOptions};
use crate::input::resolve;
use crate::metrics::reock::tally_and_save_reock;

#[derive(clap::Args, Debug)]
pub struct ReockArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    #[arg(long)]
    pub geometry_file: String,

    #[arg(long)]
    pub geometry_column: Option<String>,

    #[arg(long)]
    pub allow_geographic_crs: bool,

    #[arg(long)]
    pub allow_unknown_crs: bool,

    #[arg(long)]
    pub target_crs: Option<String>,

    #[arg(long)]
    pub source_crs: Option<String>,

    /// Stop after this many expanded samples.
    #[arg(long)]
    pub max_samples: Option<usize>,

    /// Use Brotli compression for Parquet output (default: Snappy).
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
}

pub fn run(args: ReockArgs, show_progress: bool) -> crate::error::Result<()> {
    let resolved = resolve(args.common.ben_file())?;

    let reock_geometries = load_reock_units_from_geoparquet(
        &args.geometry_file,
        ReockLoadOptions {
            geometry_column: args.geometry_column.as_deref(),
            source_crs: args.source_crs.as_deref(),
            target_crs: args.target_crs.as_deref(),
            allow_geographic_crs: args.allow_geographic_crs,
            allow_unknown_crs: args.allow_unknown_crs,
        },
    )?;

    let output_path = build_output_path(
        &args.common.ben_file(),
        "_reock.parquet",
        args.common.output_dir(),
    );

    tally_and_save_reock(
        reock_geometries,
        &resolved.source,
        &output_path,
        show_progress,
        args.max_samples,
        args.high_compression,
    )
}
