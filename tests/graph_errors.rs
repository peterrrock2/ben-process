#[path = "common/mod.rs"]
mod common;

use common::*;

#[test]
fn graph_backed_modes_require_graph_file_argument() {
    let f = fixture(&tri_plans());
    let stderr = run_failure(&[
        "--mode",
        "cut-edges",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    assert!(
        stderr.contains("graph file required; pass --graph-file <PATH>"),
        "stderr should explain missing graph argument, got: {stderr}"
    );
}

#[test]
fn graph_backed_modes_report_missing_graph_file_path() {
    let f = fixture(&tri_plans());
    let missing_graph = f.dir.join("missing_graph.json");
    let stderr = run_failure(&[
        "--mode",
        "cut-edges",
        "--graph-file",
        missing_graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    assert!(
        stderr.contains("failed to open graph file"),
        "stderr should explain graph open failure, got: {stderr}"
    );
    assert!(
        stderr.contains("missing_graph.json"),
        "stderr should include graph path, got: {stderr}"
    );
}

#[test]
fn tally_keys_reports_missing_graph_attribute_key() {
    let f = fixture(&tri_plans());
    let stderr = run_failure(&[
        "--mode",
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "does_not_exist",
        "--no-progress",
    ]);

    assert!(
        stderr.contains("failed to load numeric graph key")
            && stderr.contains("does_not_exist")
            && stderr.contains("node 0"),
        "stderr should identify the missing graph key, got: {stderr}"
    );
}

#[test]
fn graph_backed_modes_report_assignment_length_mismatch() {
    let f = fixture(&[vec![1u16, 1, 2, 2]]);
    let stderr = run_failure(&[
        "--mode",
        "cut-edges",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    assert!(
        stderr.contains("BEN assignment has 4 entries but graph has 6 nodes"),
        "stderr should explain assignment length mismatch, got: {stderr}"
    );
}
