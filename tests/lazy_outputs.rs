//! Output files are created lazily: nothing is written to disk until the first assignment decodes
//! successfully (or, for a zero-frame run, until the run completes — see `empty_inputs.rs`). These
//! tests pin the failure side of that contract: a run that dies before producing data must leave
//! no output file (and no tallies directory) behind to confuse a later audit.

#[path = "common/mod.rs"]
mod common;

use common::*;

/// 4-entry plans against the 6-node ring graph: the pipeline's assignment-length check fails on
/// the very first frame, before any output file may be created.
fn short_plan() -> Vec<Vec<u16>> {
    vec![vec![1, 1, 2, 2]]
}

#[test]
fn failed_cut_edges_run_leaves_no_output_file() {
    let f = fixture(&short_plan());
    let stderr = run_failure(&[
        "cut-edges",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "-q",
    ]);
    assert!(
        stderr.contains("BEN assignment has 4 entries but graph has 6 nodes"),
        "unexpected stderr: {stderr}"
    );
    assert!(
        !f.dir.join("plans_cut_edges.parquet").exists(),
        "failed run must not leave an output file behind"
    );
}

#[test]
fn failed_tally_keys_run_leaves_no_output_dir() {
    let f = fixture(&short_plan());
    run_failure(&[
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
        !f.dir.join("plans_tallies").exists(),
        "failed run must not leave the tallies directory behind"
    );
}

#[test]
fn failed_region_splits_run_leaves_no_output_file() {
    let f = fixture(&short_plan());
    run_failure(&[
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
        !f.dir.join("plans_region_splits.parquet").exists(),
        "failed run must not leave an output file behind"
    );
}

#[test]
fn failed_polsby_popper_run_leaves_no_output_file() {
    // 6-entry plan against the 4-node polsby path graph.
    let f = polsby_fixture(&[vec![1, 1, 2, 2, 2, 2]]);
    run_failure(&[
        "polsby-popper",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--area-key",
        "area",
        "--perim-key",
        "perim",
        "--shared-perim-key",
        "shared_perim",
        "-q",
    ]);
    assert!(
        !f.dir.join("plans_polsby_popper.parquet").exists(),
        "failed run must not leave an output file behind"
    );
}

#[test]
fn failed_unique_plans_run_leaves_no_output_file() {
    // Mixed assignment lengths within one BEN file: the second frame fails the per-file length
    // check; unique-plans only writes its single-row output after a clean full pass.
    let f = fixture(&[vec![1, 1, 2, 2, 1, 1], vec![1, 1, 2, 2]]);
    run_failure(&[
        "unique-plans",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "-q",
    ]);
    assert!(
        !f.dir.join("plans_unique_plans.parquet").exists(),
        "failed run must not leave an output file behind"
    );
}
