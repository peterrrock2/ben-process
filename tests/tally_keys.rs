#[path = "common/mod.rs"]
mod common;

use common::*;

#[test]
fn tally_keys_requires_at_least_one_key() {
    let f = fixture(&tri_plans());
    let stderr = run_failure(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "-q",
    ]);

    assert!(
        stderr.contains("at least one key is required for tally-keys mode"),
        "stderr should explain empty key list, got: {stderr}"
    );
}

#[test]
fn tally_keys_pop_per_district() {
    let f = fixture(&tri_plans());
    run(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "-q",
    ]);
    let df = read_parquet(&f.dir.join("plans_tallies").join("pop_tally_plans.parquet"));
    assert_eq!(f64_col(&df, "district_1"), vec![60.0, 90.0, 140.0]);
    assert_eq!(f64_col(&df, "district_2"), vec![150.0, 120.0, 70.0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 2, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![1, 1, 1]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1, 2, 3]);
    assert_eq!(
        df.get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        vec![
            "step".to_string(),
            "n_reps".to_string(),
            "accepted_count".to_string(),
            "district_1".to_string(),
            "district_2".to_string(),
        ]
    );
}

#[test]
fn tally_keys_multiple_keys_write_separate_files() {
    let f = fixture(&tri_plans());
    run(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "area",
        "-q",
    ]);

    let pop_df = read_parquet(&f.dir.join("plans_tallies").join("pop_tally_plans.parquet"));
    let area_df = read_parquet(&f.dir.join("plans_tallies").join("area_tally_plans.parquet"));

    assert_eq!(f64_col(&pop_df, "district_1"), vec![60.0, 90.0, 140.0]);
    assert_eq!(f64_col(&pop_df, "district_2"), vec![150.0, 120.0, 70.0]);
    assert_eq!(f64_col(&area_df, "district_1"), vec![6.0, 9.0, 14.0]);
    assert_eq!(f64_col(&area_df, "district_2"), vec![15.0, 12.0, 7.0]);
    assert_eq!(u64_col(&area_df, "step"), vec![1, 2, 3]);
    assert_eq!(u64_col(&area_df, "accepted_count"), vec![1, 2, 3]);
}

#[test]
fn tally_keys_twodelta_multiple_keys_write_separate_files() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_fixture_graph(&graph);
    write_fixture_ben_variant(&ben, &tri_plans(), BenVariant::TwoDelta);

    run(&[
        "tally-keys",
        "--graph-file",
        graph.to_str().unwrap(),
        "--ben-file",
        ben.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "--keys",
        "pop",
        "area",
        "-q",
    ]);

    let pop_df = read_parquet(&dir.join("plans_tallies").join("pop_tally_plans.parquet"));
    let area_df = read_parquet(&dir.join("plans_tallies").join("area_tally_plans.parquet"));

    assert_eq!(f64_col(&pop_df, "district_1"), vec![60.0, 90.0, 140.0]);
    assert_eq!(f64_col(&pop_df, "district_2"), vec![150.0, 120.0, 70.0]);
    assert_eq!(f64_col(&area_df, "district_1"), vec![6.0, 9.0, 14.0]);
    assert_eq!(f64_col(&area_df, "district_2"), vec![15.0, 12.0, 7.0]);
    assert_eq!(u64_col(&area_df, "step"), vec![1, 2, 3]);
    assert_eq!(u64_col(&area_df, "accepted_count"), vec![1, 2, 3]);
}

#[test]
fn tally_keys_output_dir_nests_files_under_graph_stem_directory() {
    let f = fixture(&tri_plans());
    let output_dir = f.dir.join("custom_out");
    run(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        output_dir.to_str().unwrap(),
        "--keys",
        "pop",
        "-q",
    ]);

    let expected = output_dir
        .join("plans_tallies")
        .join("pop_tally_plans.parquet");
    assert!(expected.exists(), "expected tally file at {:?}", expected);
    assert!(
        !f.dir
            .join("plans_tallies")
            .join("pop_tally_plans.parquet")
            .exists(),
        "tally file should respect --output-dir rather than defaulting to fixture dir"
    );
}

/// The district label set must be identical across every plan in the ensemble. A later plan that
/// *drops* a district present in the first plan (here district 2 vanishes) breaks the fixed
/// per-district schema just as much as one that *adds* a district, so it must fail fast rather than
/// silently emit a null/zero column.
///
/// pop fixture: plan 0 = [1,1,1,2,2,2] establishes districts {1,2}; plan 1 = [1,1,1,1,1,1] has only
/// district 1, so district 2 is missing.
#[test]
fn tally_keys_fails_when_later_frame_drops_a_first_assignment_district() {
    let plans = vec![vec![1u16, 1, 1, 2, 2, 2], vec![1u16, 1, 1, 1, 1, 1]];
    let f = fixture(&plans);
    let stderr = run_failure(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "-q",
    ]);

    assert!(
        stderr.contains("districts [2] from the first assignment are missing from a later plan")
            && stderr.contains("same district labels"),
        "stderr should explain the dropped-district schema failure, got: {stderr}"
    );
}

#[test]
fn tally_keys_fails_when_later_frames_introduce_unseen_district_ids() {
    let plans = vec![vec![1u16, 1, 1, 1, 1, 1], vec![1u16, 2, 1, 2, 1, 2]];
    let f = fixture(&plans);
    let output = Command::new(bin())
        .args([
            "tally-keys",
            "--graph-file",
            f.graph.to_str().unwrap(),
            "--ben-file",
            f.ben.to_str().unwrap(),
            "--output-dir",
            f.dir.to_str().unwrap(),
            "--keys",
            "pop",
            "-q",
        ])
        .output()
        .expect("failed to spawn ben-process");

    assert!(
        !output.status.success(),
        "tally-keys should fail on unseen district ids"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("not present in first assignment"),
        "stderr should explain the streaming-schema failure, got: {stderr}"
    );
}

/// MkvChain BEN: `step` advances by `n_reps`, `accepted_count` by 1, and per-district tallies are
/// emitted once per frame (not once per repeated sample).
///
/// Frames after coalescing:
///   frame 1: [1,1,1,2,2,2], count=2 → pop d1=60,  d2=150
///   frame 2: [1,2,1,2,1,2], count=1 → pop d1=90,  d2=120
#[test]
fn tally_keys_mkvchain_step_advances_by_n_reps() {
    let f = fixture_mkv(&[
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 1, 1, 2, 2, 2], // coalesces with previous → count=2
        vec![1, 2, 1, 2, 1, 2],
    ]);
    run(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "-q",
    ]);
    let df = read_parquet(&f.dir.join("plans_tallies").join("pop_tally_plans.parquet"));
    assert_eq!(f64_col(&df, "district_1"), vec![60.0, 90.0]);
    assert_eq!(f64_col(&df, "district_2"), vec![150.0, 120.0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![2, 1]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1, 2]);
}

#[test]
fn tally_keys_max_samples_truncates_mkvchain_repetition_count() {
    let f = fixture_mkv(&[
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 2, 1, 2, 1, 2],
    ]);
    run(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "--max-samples",
        "2",
        "-q",
    ]);
    let df = read_parquet(
        &f.dir
            .join("plans_tallies")
            .join("pop_tally_up_to_2_plans.parquet"),
    );
    assert_eq!(f64_col(&df, "district_1"), vec![60.0]);
    assert_eq!(f64_col(&df, "district_2"), vec![150.0]);
    assert_eq!(u64_col(&df, "step"), vec![1]);
    assert_eq!(u32_col(&df, "n_reps"), vec![2]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1]);
}

#[test]
fn tally_keys_twodelta_max_samples_truncates_repetition_count() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_fixture_graph(&graph);
    write_fixture_ben_variant(
        &ben,
        &[
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 2, 1, 2, 1, 2],
        ],
        BenVariant::TwoDelta,
    );

    run(&[
        "tally-keys",
        "--graph-file",
        graph.to_str().unwrap(),
        "--ben-file",
        ben.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "--keys",
        "pop",
        "--max-samples",
        "2",
        "-q",
    ]);

    let df = read_parquet(
        &dir.join("plans_tallies")
            .join("pop_tally_up_to_2_plans.parquet"),
    );
    assert_eq!(f64_col(&df, "district_1"), vec![60.0]);
    assert_eq!(f64_col(&df, "district_2"), vec![150.0]);
    assert_eq!(u64_col(&df, "step"), vec![1]);
    assert_eq!(u32_col(&df, "n_reps"), vec![2]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1]);
}

#[test]
fn tally_keys_max_samples_reports_short_input() {
    let f = fixture(&tri_plans());
    let stderr = run_success_capture_stderr(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "--max-samples",
        "5",
        // The short-input notice is an info-level log line, off by default; `-v` enables it.
        "-v",
        "-q",
    ]);

    assert!(
        stderr.contains("Reached end of input after 3 samples before --max-samples 5"),
        "stderr should report the short input, got: {stderr}"
    );
}

/// Duplicate keys would derive the same output path twice: two writers interleaving row groups
/// into one file produces unreadable Parquet, so the CLI must reject the duplicate up front.
#[test]
fn tally_keys_rejects_duplicate_keys() {
    let f = fixture(&tri_plans());
    let stderr = run_failure(&[
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "pop",
        "-q",
    ]);
    assert!(
        stderr.contains("duplicate key \"pop\" passed to --keys"),
        "stderr should report the duplicate key, got: {stderr}"
    );
}
