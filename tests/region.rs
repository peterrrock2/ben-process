#[path = "common/mod.rs"]
mod common;

use common::*;

#[test]
fn region_splits_for_region_key() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode",
        "region-splits",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "region",
        "--no-progress",
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
        "--mode",
        "region-splits",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "region",
        "--no-progress",
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
        "--mode",
        "region-pieces",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "region",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_region_pieces.parquet"));
    assert_eq!(u32_col(&df, "region_pieces"), vec![4, 4, 2]);
}
