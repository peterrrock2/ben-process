#[path = "common/mod.rs"]
mod common;

use common::*;

#[test]
fn region_splits_for_region_key() {
    let f = fixture(&tri_plans());
    run(&[
        "region-splits",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "region",
        "-q",
    ]);
    let df = read_parquet(&f.dir.join("plans_region_splits.parquet"));
    assert_eq!(u32_col(&df, "region_splits"), vec![2, 2, 0]);
    assert_eq!(
        str_col(&df, "region_key"),
        vec!["region", "region", "region"]
    );
}

/// Region modes also go through the pipeline's fixed-district-set chokepoint. plan 0 =
/// [1,1,1,2,2,2] establishes districts {1,2}; plan 1 = [1,1,1,1,1,1] drops district 2 → fail fast.
#[test]
fn region_splits_fails_when_district_set_changes() {
    let plans = vec![vec![1u16, 1, 1, 2, 2, 2], vec![1u16, 1, 1, 1, 1, 1]];
    let f = fixture(&plans);
    let stderr = run_failure(&[
        "region-splits",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "region",
        "-q",
    ]);

    assert!(
        stderr.contains("districts [2] from the first assignment are missing from a later plan")
            && stderr.contains("same district labels"),
        "stderr should explain the changed-district-set failure, got: {stderr}"
    );
}

#[test]
fn region_pieces_for_region_key() {
    let f = fixture(&tri_plans());
    run(&[
        "region-pieces",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "region",
        "-q",
    ]);
    let df = read_parquet(&f.dir.join("plans_region_pieces.parquet"));
    assert_eq!(u32_col(&df, "region_pieces"), vec![4, 4, 2]);
}

/// MkvChain BEN: `step` advances by `n_reps`, `accepted_count` by 1, and the region metric is
/// emitted once per frame.
///
/// Frames after coalescing (region = A,A,B,B,A,A):
///   frame 1: [1,1,1,2,2,2], count=2 → A={1,2}, B={1,2} → splits=2
///   frame 2: [1,1,2,2,1,1], count=1 → A={1},   B={2}   → splits=0
#[test]
fn region_splits_mkvchain_step_advances_by_n_reps() {
    let f = fixture_mkv(&[
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 1, 1, 2, 2, 2], // coalesces with previous → count=2
        vec![1, 1, 2, 2, 1, 1],
    ]);
    run(&[
        "region-splits",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "region",
        "-q",
    ]);
    let df = read_parquet(&f.dir.join("plans_region_splits.parquet"));
    assert_eq!(u32_col(&df, "region_splits"), vec![2, 0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![2, 1]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1, 2]);
}
