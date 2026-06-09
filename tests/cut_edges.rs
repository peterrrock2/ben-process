#[path = "common/mod.rs"]
mod common;

use common::*;

#[test]
fn cut_edges_unweighted() {
    let f = fixture(&tri_plans());
    run(&[
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
    let df = read_parquet(&f.dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![2.0, 6.0, 2.0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 2, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![1, 1, 1]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1, 2, 3]);
}

#[test]
fn cut_edges_weighted_by_edge_key() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode",
        "cut-edges",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--edge-weight-key",
        "weight",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![8.0, 21.0, 5.0]);
}

/// Asymmetric edge-weight fixture: the same edge carries a valid weight from one endpoint and a
/// missing/non-numeric value from the other. The edge-weight rule is "last valid weight wins" (a
/// non-numeric value is not stored), so cut_edges on p0=[1,1,1,2,2,2] must give 8.0 regardless of
/// which endpoint's JSON entry is seen first.
///
/// Edge (0,5): node 0 says weight missing; node 5 says weight=3.0 → must pick 3.0. Edge (2,3): node
/// 2 says weight=5.0; node 3 says weight missing → must pick 5.0. p0 cuts (2,3) and (0,5), so
/// cut_edges = 5.0 + 3.0 = 8.0.
#[test]
fn cut_edges_weighted_tolerates_asymmetric_missing_weight() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    // Edge (0,5): weight only on node 5's side. Edge (2,3): weight only on node 2's side. All
    // others symmetric with weight 1.0.
    let graph_json = serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "pop": 10.0, "region": "A" },
            { "pop": 20.0, "region": "A" },
            { "pop": 30.0, "region": "B" },
            { "pop": 40.0, "region": "B" },
            { "pop": 50.0, "region": "A" },
            { "pop": 60.0, "region": "A" },
        ],
        "adjacency": [
            [ { "id": 1, "weight": 1.0 }, { "id": 5 } ],
            [ { "id": 0, "weight": 1.0 }, { "id": 2, "weight": 1.0 } ],
            [ { "id": 1, "weight": 1.0 }, { "id": 3, "weight": 5.0 } ],
            [ { "id": 2 },                { "id": 4, "weight": 1.0 } ],
            [ { "id": 3, "weight": 1.0 }, { "id": 5, "weight": 1.0 } ],
            [ { "id": 0, "weight": 3.0 }, { "id": 4, "weight": 1.0 } ],
        ]
    });
    std::fs::write(&graph, graph_json.to_string()).unwrap();
    write_fixture_ben(&ben, &[vec![1u16, 1, 1, 2, 2, 2]]);
    run(&[
        "--mode",
        "cut-edges",
        "--graph-file",
        graph.to_str().unwrap(),
        "--ben-file",
        ben.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "--edge-weight-key",
        "weight",
        "--no-progress",
    ]);
    let df = read_parquet(&dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![8.0]);
}

/// `--high-compression` switches the parquet writer from Snappy to Brotli. None of the other
/// snapshot tests exercise this branch, so a regression in the Brotli writer setup wouldn't be
/// caught. Run cut-edges with --high-compression and verify the output is still a valid parquet
/// file with the expected values; the polars reader is compression-agnostic.
#[test]
fn cut_edges_with_high_compression_round_trips() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode",
        "cut-edges",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--high-compression",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![2.0, 6.0, 2.0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 2, 3]);
}

/// Cut-edges has no per-district output schema, but the fixed-district-set invariant is now
/// enforced centrally in the pipeline (cut-edges captures the set from the edge endpoints it
/// already walks). plan 0 = [1,1,1,2,2,2] establishes districts {1,2}; plan 1 = [1,1,1,1,1,1] drops
/// district 2, so the run must fail fast.
#[test]
fn cut_edges_fails_when_district_set_changes() {
    let plans = vec![vec![1u16, 1, 1, 2, 2, 2], vec![1u16, 1, 1, 1, 1, 1]];
    let f = fixture(&plans);
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
        stderr.contains("districts [2] from the first assignment are missing from a later plan")
            && stderr.contains("same district labels"),
        "stderr should explain the changed-district-set failure, got: {stderr}"
    );
}

/// `--output-dir` pointing at an existing *file* (not a directory) cannot host the output parquet.
/// `File::create` will fail; the binary must surface a non-zero exit rather than silently producing
/// nothing.
#[test]
fn cut_edges_fails_when_output_dir_is_an_existing_file() {
    let f = fixture(&tri_plans());
    let bogus_dir = f.dir.join("not_a_dir");
    std::fs::write(&bogus_dir, b"i am a regular file").unwrap();

    let output = Command::new(bin())
        .args([
            "--mode",
            "cut-edges",
            "--graph-file",
            f.graph.to_str().unwrap(),
            "--ben-file",
            f.ben.to_str().unwrap(),
            "--output-dir",
            bogus_dir.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .expect("failed to spawn ben-process");

    assert!(
        !output.status.success(),
        "cut-edges should fail when --output-dir is an existing file; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// MkvChain BEN coalesces consecutive identical assignments into one frame with `count > 1`.
/// Through `run_pipeline` that means `step` advances by `n_reps` while `accepted_count` advances by
/// 1, and the `n_reps` column reflects the repetition count. Standard-BEN tests only ever see
/// `n_reps == 1`.
///
/// Frames after run-length coalescing:
///   frame 1: [1,1,1,2,2,2], count=2 → cut_edges=2
///   frame 2: [1,2,1,2,1,2], count=1 → cut_edges=6
#[test]
fn cut_edges_mkvchain_step_advances_by_n_reps() {
    let f = fixture_mkv(&[
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 1, 1, 2, 2, 2], // coalesces with previous → count=2
        vec![1, 2, 1, 2, 1, 2],
    ]);
    run(&[
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
    let df = read_parquet(&f.dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![2.0, 6.0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![2, 1]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1, 2]);
}
