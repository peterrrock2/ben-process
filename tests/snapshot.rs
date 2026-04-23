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

/// Smoke test for changed-assignments on a single-plan file. The full
/// multi-plan case depends on a CLI flag (`--mkv-rand-reassignment-off`) whose
/// default semantics are inverted from its name, making multi-plan runs
/// nondeterministic. We fix that in a later phase; for now verify the mode
/// runs end-to-end and emits the expected all-zeros output for one plan.
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
