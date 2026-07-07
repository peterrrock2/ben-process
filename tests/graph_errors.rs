#[path = "common/mod.rs"]
mod common;

use common::*;

#[test]
fn rejects_ben_file_path_without_file_name() {
    // A --ben-file with no file-name component (here "..") would panic in ben_stem when deriving
    // the output path; the entry-point guard must turn it into a clean error instead.
    // unique-plans needs no graph, so this isolates the ben-file check.
    let stderr = run_failure(&["unique-plans", "--ben-file", "..", "-q"]);

    assert!(
        stderr.contains("path has no file name component"),
        "stderr should explain the invalid --ben-file, got: {stderr}"
    );
}

#[test]
fn graph_backed_modes_require_graph_file_argument() {
    let f = fixture(&tri_plans());
    let stderr = run_failure(&[
        "cut-edges",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "-q",
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
        "cut-edges",
        "--graph-file",
        missing_graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "-q",
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
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "does_not_exist",
        "-q",
    ]);

    assert!(
        stderr.contains("failed to load numeric graph key")
            && stderr.contains("does_not_exist")
            && stderr.contains("node 0"),
        "stderr should identify the missing graph key, got: {stderr}"
    );
}

/// A graph whose node ids don't match their `.nodes[]` positions must still load correctly —
/// `.nodes[]` order is the true node order, and adjacency ids are resolved through the `id`
/// labels — while warning the user about the mismatch.
///
/// The fixture is a 4-node path (by position: 0 - 1 - 2 - 3) with reversed ids [3, 2, 1, 0] and
/// adjacency referencing those ids. With the plan [1, 1, 2, 2] the true cut-edge count is 1 (only
/// the positional edge (1, 2) crosses). Misreading adjacency ids as positions would instead build
/// edges {(0, 2), (1, 3)} and report 2 — so the asserted value pins the id resolution, not just
/// the absence of an error.
#[test]
fn permuted_node_ids_warn_and_resolve_against_nodes_order() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");

    let graph_json = serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "id": 3 },
            { "id": 2 },
            { "id": 1 },
            { "id": 0 },
        ],
        "adjacency": [
            [ { "id": 2 } ],
            [ { "id": 3 }, { "id": 1 } ],
            [ { "id": 2 }, { "id": 0 } ],
            [ { "id": 1 } ]
        ]
    });
    std::fs::write(&graph, graph_json.to_string()).unwrap();
    write_fixture_ben(&ben, &[vec![1u16, 1, 2, 2]]);

    let stderr = run_success_capture_stderr(&[
        "cut-edges",
        "--graph-file",
        graph.to_str().unwrap(),
        "--ben-file",
        ben.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);

    assert!(
        stderr.contains("graph node ids do not match their positions"),
        "stderr should warn about the id/position mismatch, got: {stderr}"
    );

    let df = read_parquet(&dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![1.0]);
}

#[test]
fn graph_backed_modes_report_assignment_length_mismatch() {
    let f = fixture(&[vec![1u16, 1, 2, 2]]);
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
        stderr.contains("BEN assignment length is 4 but graph node count is 6"),
        "stderr should explain assignment length mismatch, got: {stderr}"
    );
}
