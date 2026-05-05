//! End-to-end regression tests for ben-tally.
//!
//! Each test builds a tiny 6-node ring fixture + a handful of assignment vectors,
//! invokes the compiled `ben-tally` binary via `env!("CARGO_BIN_EXE_ben-tally")`,
//! and asserts the produced Parquet / text output against manually-computed
//! expected values. The fixture is intentionally small enough that a reader can
//! verify every expected value on paper.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use ben::decode::BenDecoder;
use ben::encode::BenEncoder;
use ben::BenVariant;
use polars::prelude::*;
use tempfile::{tempdir, TempDir};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_ben-tally")
}

// Six-node ring:
//     0 - 1 - 2 - 3 - 4 - 5 - 0
// with "pop" = 10 * (idx + 1) and "region" = A/A/B/B/A/A.
// Edge "weight" varies per edge so we can exercise --edge-weight-key.
fn write_fixture_graph(path: &Path) {
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
            [ { "id": 1, "weight": 2.0 }, { "id": 5, "weight": 3.0 } ],
            [ { "id": 0, "weight": 2.0 }, { "id": 2, "weight": 1.0 } ],
            [ { "id": 1, "weight": 1.0 }, { "id": 3, "weight": 5.0 } ],
            [ { "id": 2, "weight": 5.0 }, { "id": 4, "weight": 4.0 } ],
            [ { "id": 3, "weight": 4.0 }, { "id": 5, "weight": 6.0 } ],
            [ { "id": 0, "weight": 3.0 }, { "id": 4, "weight": 6.0 } ],
        ]
    });
    let mut f = File::create(path).unwrap();
    f.write_all(graph_json.to_string().as_bytes()).unwrap();
}

fn write_fixture_ben(path: &Path, plans: &[Vec<u16>]) {
    let f = File::create(path).unwrap();
    let mut enc = BenEncoder::new(f, BenVariant::Standard);
    for p in plans {
        enc.write_assignment(p.clone()).unwrap();
    }
    enc.finish().unwrap();
}

/// Encode with `BenVariant::MkvChain` so consecutive identical assignments
/// collapse into a single frame with `count > 1`. Used by the MkvChain
/// regression test for changed-assignments frame counting.
fn write_fixture_ben_mkv(path: &Path, plans: &[Vec<u16>]) {
    let f = File::create(path).unwrap();
    let mut enc = BenEncoder::new(f, BenVariant::MkvChain);
    for p in plans {
        enc.write_assignment(p.clone()).unwrap();
    }
    enc.finish().unwrap();
}

struct Fixture {
    _tmp: TempDir, // keeps the temp dir alive for the duration of the test
    dir: PathBuf,
    graph: PathBuf,
    ben: PathBuf,
}

fn fixture(plans: &[Vec<u16>]) -> Fixture {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_fixture_graph(&graph);
    write_fixture_ben(&ben, plans);
    Fixture { _tmp: tmp, dir, graph, ben }
}

fn run(args: &[&str]) {
    let status = Command::new(bin())
        .args(args)
        .status()
        .expect("failed to spawn ben-tally");
    assert!(status.success(), "ben-tally exited non-zero");
}

fn read_parquet(path: &Path) -> DataFrame {
    ParquetReader::new(&mut File::open(path).unwrap())
        .finish()
        .unwrap()
}

fn f64_col(df: &DataFrame, name: &str) -> Vec<f64> {
    df.column(name)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

fn u32_col(df: &DataFrame, name: &str) -> Vec<u32> {
    df.column(name)
        .unwrap()
        .u32()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

fn u64_col(df: &DataFrame, name: &str) -> Vec<u64> {
    df.column(name)
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

fn str_col(df: &DataFrame, name: &str) -> Vec<String> {
    df.column(name)
        .unwrap()
        .str()
        .unwrap()
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect()
}

/// Three plans used by the main per-mode tests. Manually verified values:
///
///  pop per district (tally-keys):
///    p0=[1,1,1,2,2,2]  d1=60,  d2=150
///    p1=[1,2,1,2,1,2]  d1=90,  d2=120
///    p2=[1,1,2,2,1,1]  d1=140, d2=70
///
///  unweighted cut_edges: 2, 6, 2
///  weighted cut_edges:   8, 21, 5
///
///  region "region" per-sample districts-in-region table:
///    p0=[1,1,1,2,2,2]: A={1,1,2,2}->{1,2}, B={1,2}->{1,2}         splits=2 pieces=4
///    p1=[1,2,1,2,1,2]: A={1,2,1,2}->{1,2}, B={1,2}->{1,2}         splits=2 pieces=4
///    p2=[1,1,2,2,1,1]: A={1,1,1,1}->{1},   B={2,2}->{2}           splits=0 pieces=2
fn tri_plans() -> Vec<Vec<u16>> {
    vec![
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 2, 1, 2, 1, 2],
        vec![1, 1, 2, 2, 1, 1],
    ]
}

#[test]
fn cut_edges_unweighted() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode", "cut-edges",
        "--graph-file", f.graph.to_str().unwrap(),
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![2.0, 6.0, 2.0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 2, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![1, 1, 1]);
    assert_eq!(u32_col(&df, "accepted_count"), vec![1, 2, 3]);
}

#[test]
fn cut_edges_weighted_by_edge_key() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode", "cut-edges",
        "--graph-file", f.graph.to_str().unwrap(),
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--edge-weight-key", "weight",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![8.0, 21.0, 5.0]);
}

#[test]
fn tally_keys_pop_per_district() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode", "tally-keys",
        "--graph-file", f.graph.to_str().unwrap(),
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--keys", "pop",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_tallies.parquet"));
    assert_eq!(f64_col(&df, "district_1"), vec![60.0, 90.0, 140.0]);
    assert_eq!(f64_col(&df, "district_2"), vec![150.0, 120.0, 70.0]);
    assert_eq!(str_col(&df, "sum_columns"), vec!["pop", "pop", "pop"]);
    assert_eq!(u64_col(&df, "step"), vec![1, 2, 3]);
    // Stable column order after the BTreeMap fix: step / n_reps / accepted_count
    // / sum_columns / district_1 / district_2.
    assert_eq!(
        df.get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        vec![
            "step".to_string(),
            "n_reps".to_string(),
            "accepted_count".to_string(),
            "sum_columns".to_string(),
            "district_1".to_string(),
            "district_2".to_string(),
        ]
    );
}

#[test]
fn region_splits_for_region_key() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode", "region-splits",
        "--graph-file", f.graph.to_str().unwrap(),
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--keys", "region",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_region_splits.parquet"));
    assert_eq!(u32_col(&df, "region_splits"), vec![2, 2, 0]);
    assert_eq!(str_col(&df, "region_key"), vec!["region", "region", "region"]);
}

#[test]
fn region_pieces_for_region_key() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode", "region-pieces",
        "--graph-file", f.graph.to_str().unwrap(),
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--keys", "region",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_region_pieces.parquet"));
    assert_eq!(u32_col(&df, "region_pieces"), vec![4, 4, 2]);
}

#[test]
fn changed_assignments_single_plan_smoke() {
    let plans = vec![vec![1u16, 1, 1, 2, 2, 2]];
    let f = fixture(&plans);
    run(&[
        "--mode", "changed-assignments",
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let body = std::fs::read_to_string(
        f.dir.join("plans_accept_1_changed_assignments.txt"),
    )
    .unwrap();
    assert_eq!(body, "[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\nTotal Accepted: 1");
}

/// Multi-plan changed-assignments: with `--randomize-reassignments` default
/// `false`, output is deterministic.
///
/// Manual trace for `tri_plans()`:
///  - curr=[1,1,1,2,2,2] vs p1=[1,2,1,2,1,2] → diffs at i=1,4  → dif=[0,1,0,0,1,0]
///  - curr=[1,2,1,2,1,2] vs p2=[1,1,2,2,1,1] → diffs at i=1,2,5 → dif=[0,2,1,0,1,1]
#[test]
fn changed_assignments_tri_plans_deterministic() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode", "changed-assignments",
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let body = std::fs::read_to_string(
        f.dir.join("plans_accept_3_changed_assignments.txt"),
    )
    .unwrap();
    assert_eq!(body, "[0.0, 2.0, 1.0, 0.0, 1.0, 1.0]\nTotal Accepted: 3");
}

/// MkvChain BEN with a repeated assignment collapses into a frame with
/// `count > 1`. `changed-assignments` semantics are per-accepted-record
/// (per-frame), so a 3-sample / 2-frame ensemble should report 2 accepted.
///
/// Fixture frames (after MkvChain run-length):
///   frame 1: assignment=[1,1,1,2,2,2], count=2 (two repeated samples)
///   frame 2: assignment=[1,2,1,2,1,2], count=1
///
/// With per-frame semantics:
///   - curr=[1,1,1,2,2,2] (first frame)
///   - curr vs [1,2,1,2,1,2] → diffs at i=1, i=4 → dif_count=[0,1,0,0,1,0]
///   - Output filename carries "_accept_2_" (frames), not "_accept_3_".
#[test]
fn changed_assignments_mkvchain_uses_frame_count() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_fixture_graph(&graph);
    write_fixture_ben_mkv(
        &ben,
        &[
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 1, 1, 2, 2, 2], // coalesces with previous → count=2
            vec![1, 2, 1, 2, 1, 2],
        ],
    );
    run(&[
        "--mode", "changed-assignments",
        "--ben-file", ben.to_str().unwrap(),
        "--output-dir", dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let body = std::fs::read_to_string(dir.join("plans_accept_2_changed_assignments.txt"))
        .expect("output file should use frame count (2), not sample count (3)");
    assert_eq!(body, "[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]\nTotal Accepted: 2");
}

/// Asymmetric edge-weight fixture: the same edge carries a valid weight from
/// one endpoint and a missing/non-numeric value from the other. The
/// pre-refactor code's semantics were "last valid weight wins" (and don't
/// store anything when the parsed value isn't numeric). Must still give 8.0
/// for cut_edges on p0=[1,1,1,2,2,2] regardless of which endpoint's JSON
/// entry is seen first.
///
/// Edge (0,5): node 0 says weight missing; node 5 says weight=3.0 → must pick 3.0.
/// Edge (2,3): node 2 says weight=5.0; node 3 says weight missing → must pick 5.0.
/// p0 cuts (2,3) and (0,5), so cut_edges = 5.0 + 3.0 = 8.0.
#[test]
fn cut_edges_weighted_tolerates_asymmetric_missing_weight() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    // Edge (0,5): weight only on node 5's side.
    // Edge (2,3): weight only on node 2's side.
    // All others symmetric with weight 1.0.
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
        "--mode", "cut-edges",
        "--graph-file", graph.to_str().unwrap(),
        "--ben-file", ben.to_str().unwrap(),
        "--output-dir", dir.to_str().unwrap(),
        "--edge-weight-key", "weight",
        "--no-progress",
    ]);
    let df = read_parquet(&dir.join("plans_cut_edges.parquet"));
    assert_eq!(f64_col(&df, "cut_edges"), vec![8.0]);
}

/// Normalize divides each count by `line_count - 1` = 2.
#[test]
fn changed_assignments_tri_plans_normalized() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode", "changed-assignments",
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--normalize",
        "--no-progress",
    ]);
    let body = std::fs::read_to_string(
        f.dir.join("plans_accept_3_changed_assignments.txt"),
    )
    .unwrap();
    assert_eq!(body, "[0.0, 1.0, 0.5, 0.0, 0.5, 0.5]\nTotal Accepted: 3");
}

/// Five input frames with three label-invariant partitions:
///   * P_A appears as itself and again with districts {1,2} swapped (label-perm)
///   * P_B appears as itself and again byte-identical
///   * P_C appears once
/// Expected: extract-unique-plans writes exactly the 3 first-occurrences,
/// preserving original labels of the first time each partition was seen.
#[test]
fn extract_unique_plans_dedups_label_permutations() {
    let plans: Vec<Vec<u16>> = vec![
        vec![1, 1, 1, 2, 2, 2], // P_A first
        vec![1, 1, 2, 2, 1, 1], // P_B first
        vec![2, 2, 2, 1, 1, 1], // P_A again, labels swapped — should dedup
        vec![1, 1, 2, 2, 1, 1], // P_B again, identical — should dedup
        vec![1, 2, 1, 2, 1, 2], // P_C first
    ];
    let f = fixture(&plans);
    run(&[
        "--mode", "extract-unique-plans",
        "--ben-file", f.ben.to_str().unwrap(),
        "--output-dir", f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    let out = f.dir.join("plans_unique.jsonl.ben");
    let decoder = BenDecoder::new(File::open(&out).unwrap()).unwrap();
    let extracted: Vec<Vec<u16>> = decoder
        .map(|r| r.unwrap().0)
        .collect();

    assert_eq!(
        extracted,
        vec![
            vec![1u16, 1, 1, 2, 2, 2], // P_A first occurrence (original labels)
            vec![1u16, 1, 2, 2, 1, 1], // P_B first occurrence
            vec![1u16, 2, 1, 2, 1, 2], // P_C first occurrence
        ]
    );
}
