use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug, Clone, ValueEnum, PartialEq)]
pub enum Mode {
    TallyKeys,
    CutEdges,
    ChangedAssignments,
    RegionSplits,
    RegionPieces,
}

#[derive(Parser, Debug)]
#[command(
    name = "BEN Parquet Tally Tool",
    about = "A tool for tallying and saving data from BEN files to Parquet files.",
    version = "0.1.0"
)]
pub struct Args {
    #[arg(short, long, default_value = "cut-edges")]
    pub mode: Mode,
    #[arg(short, long)]
    pub graph_file: Option<String>,
    #[arg(short, long)]
    pub ben_file: String,
    #[arg(short, long, default_value_t = false)]
    pub normalize: bool,
    #[arg(short, long)]
    pub max_accepted: Option<usize>,
    #[arg(short, long, default_value_t = true)]
    pub mkv_rand_reassignment_off: bool,
    #[arg(short, long, num_args(1..))]
    pub keys: Vec<String>,
    #[arg(long)]
    pub edge_weight_key: Option<String>,
    #[arg(long, default_value_t = false)]
    pub no_progress: bool,
    #[arg(long)]
    pub output_dir: Option<String>,
}

pub fn build_output_path(in_ben_file: &str, suffix: &str, output_dir: Option<&str>) -> String {
    let base_name = Path::new(in_ben_file)
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy()
        .replace(".jsonl.ben", suffix);

    match output_dir {
        Some(dir) => PathBuf::from(dir)
            .join(base_name)
            .to_string_lossy()
            .into_owned(),
        _ => in_ben_file.replace(".jsonl.ben", suffix),
    }
}
