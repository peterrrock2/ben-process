use std::path::{Path, PathBuf};

/// Build an output path next to (or alongside) `in_ben_file` by stripping the BEN extension and
/// appending `suffix`.
///
/// The previous implementation used `String::replace(".jsonl.ben", suffix)`, which silently
/// returned the input unchanged when the input did not end in `.jsonl.ben` — meaning a downstream
/// `File::create` could overwrite the input BEN. This version guarantees the output path differs
/// from the input path: it strips a trailing `.jsonl.ben` (or just `.ben`) before appending, and
/// asserts distinctness as a safety net.
pub fn build_output_path(in_ben_file: &str, suffix: &str, output_dir: Option<&str>) -> String {
    let in_path = Path::new(in_ben_file);
    let stem = ben_stem(in_ben_file);
    let new_name = format!("{}{}", stem, suffix);

    let out = match output_dir {
        Some(dir) => PathBuf::from(dir)
            .join(new_name)
            .to_string_lossy()
            .into_owned(),
        None => match in_path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                parent.join(new_name).to_string_lossy().into_owned()
            }
            _ => new_name,
        },
    };

    assert!(
        out != in_ben_file,
        "refusing to overwrite input BEN file {:?}: derived output path is identical",
        in_ben_file
    );
    out
}

fn ben_stem(in_ben_file: &str) -> String {
    let in_path = Path::new(in_ben_file);
    let base_name = in_path
        .file_name()
        .expect("Failed to extract basename")
        .to_string_lossy()
        .into_owned();
    base_name
        .strip_suffix(".jsonl.ben")
        .or_else(|| base_name.strip_suffix(".ben"))
        .or_else(|| base_name.strip_suffix(".xben"))
        .or_else(|| base_name.strip_suffix(".bendl"))
        .unwrap_or(&base_name)
        .to_string()
}

pub fn build_tally_output_dir(in_ben_file: &str, output_dir: Option<&str>) -> PathBuf {
    let ben_path = Path::new(in_ben_file);
    let ben_stem = ben_stem(in_ben_file);
    let dir_name = format!("{}_tallies", ben_stem);

    match output_dir {
        Some(dir) => PathBuf::from(dir).join(dir_name),
        None => ben_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(dir_name),
    }
}

pub fn build_tally_output_path(in_ben_file: &str, key: &str, output_dir: Option<&str>) -> PathBuf {
    let ben_stem = ben_stem(in_ben_file);
    build_tally_output_dir(in_ben_file, output_dir)
        .join(format!("{}_tally_{}.parquet", key, ben_stem))
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
                "_unique_plans.parquet",
                Some("/tmp/out"),
            ),
            "/tmp/out/plans_unique_plans.parquet"
        );
    }

    #[test]
    fn build_output_path_strips_plain_ben_extension() {
        assert_eq!(
            build_output_path("/tmp/runs/plans.ben", "_cut_edges.parquet", None),
            "/tmp/runs/plans_cut_edges.parquet"
        );
    }

    #[test]
    fn build_output_path_appends_suffix_when_input_has_no_known_extension() {
        // An input with no recognized BEN extension still gets the suffix appended, so the derived
        // output path can never collide with the input file.
        assert_eq!(
            build_output_path("/tmp/runs/plans", "_cut_edges.parquet", None),
            "/tmp/runs/plans_cut_edges.parquet"
        );
    }

    #[test]
    fn build_output_path_handles_bare_filename_without_parent_dir() {
        assert_eq!(
            build_output_path("plans.jsonl.ben", "_cut_edges.parquet", None),
            "plans_cut_edges.parquet"
        );
    }

    #[test]
    #[should_panic(expected = "refusing to overwrite input BEN file")]
    fn build_output_path_panics_if_output_would_equal_input() {
        // An empty suffix combined with a no-extension input would yield the same path; the
        // assertion must catch this before File::create runs.
        let _ = build_output_path("/tmp/runs/plans", "", None);
    }

    #[test]
    fn build_tally_output_dir_uses_ben_stem() {
        assert_eq!(
            build_tally_output_dir("/tmp/runs/plans.jsonl.ben", Some("/tmp/out")),
            PathBuf::from("/tmp/out/plans_tallies")
        );
    }

    #[test]
    fn build_tally_output_path_uses_key_and_ben_stem() {
        assert_eq!(
            build_tally_output_path("/tmp/runs/plans.jsonl.ben", "pop", Some("/tmp/out")),
            PathBuf::from("/tmp/out/plans_tallies/pop_tally_plans.parquet")
        );
    }
}
