use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug, Clone, ValueEnum, PartialEq)]
pub enum Mode {
    TallyKeys,
    CutEdges,
    ChangedAssignments,
    RegionSplits,
    RegionPieces,
    UniquePlans,
    ExtractUniquePlans,
}

#[derive(Parser, Debug)]
#[command(
    name = "BEN Process Tool",
    about = "A tool for processing BEN files and saving outputs to Parquet or text.",
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
    #[arg(long)]
    pub max_accepted: Option<usize>,
    /// Randomize merge-split label reassignments (changed-assignments mode only).
    /// Only set this for MCMC merge-split ensembles. Default: off.
    #[arg(long, default_value_t = false)]
    pub randomize_reassignments: bool,
    #[arg(short, long, num_args(1..))]
    pub keys: Vec<String>,
    #[arg(long)]
    pub edge_weight_key: Option<String>,
    #[arg(long, default_value_t = false)]
    pub no_progress: bool,
    #[arg(long)]
    pub output_dir: Option<String>,
    /// Use Brotli compression for Parquet output (default: Snappy).
    /// Brotli is CPU-heavy and rarely worth it unless you're storage-bound.
    #[arg(long, default_value_t = false)]
    pub high_compression: bool,
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

pub fn build_tally_output_dir(graph_file: &str, output_dir: Option<&str>) -> PathBuf {
    let graph_path = Path::new(graph_file);
    let graph_stem = graph_path
        .file_stem()
        .expect("Failed to extract graph stem")
        .to_string_lossy();
    let dir_name = format!("{}_tallies", graph_stem);

    match output_dir {
        Some(dir) => PathBuf::from(dir).join(dir_name),
        None => graph_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(dir_name),
    }
}

pub fn build_tally_output_path(graph_file: &str, key: &str, output_dir: Option<&str>) -> PathBuf {
    let graph_path = Path::new(graph_file);
    let graph_stem = graph_path
        .file_stem()
        .expect("Failed to extract graph stem")
        .to_string_lossy();
    build_tally_output_dir(graph_file, output_dir)
        .join(format!("{}_tally_{}.parquet", key, graph_stem))
}

#[cfg(test)]
mod tests {
    use super::{build_output_path, build_tally_output_dir, build_tally_output_path};
    use std::path::PathBuf;

    #[test]
    fn build_output_path_replaces_suffix_in_place_without_output_dir() {
        assert_eq!(
            build_output_path("/tmp/runs/plans.jsonl.ben", "_cut_edges.parquet", None),
            "/tmp/runs/plans_cut_edges.parquet"
        );
    }

    #[test]
    fn build_output_path_uses_basename_when_output_dir_is_set() {
        assert_eq!(
            build_output_path(
                "/tmp/runs/plans.jsonl.ben",
                "_unique_plans.txt",
                Some("/tmp/out"),
            ),
            "/tmp/out/plans_unique_plans.txt"
        );
    }

    #[test]
    fn build_tally_output_dir_uses_graph_stem() {
        assert_eq!(
            build_tally_output_dir("/tmp/runs/graph.json", Some("/tmp/out")),
            PathBuf::from("/tmp/out/graph_tallies")
        );
    }

    #[test]
    fn build_tally_output_path_uses_key_and_graph_stem() {
        assert_eq!(
            build_tally_output_path("/tmp/runs/graph.json", "pop", Some("/tmp/out")),
            PathBuf::from("/tmp/out/graph_tallies/pop_tally_graph.parquet")
        );
    }
}
