//! End-to-end regression tests for ben-process.
//!
//! Each test builds a tiny 6-node ring fixture + a handful of assignment vectors,
//! invokes the compiled `ben-process` binary via `env!("CARGO_BIN_EXE_ben-process")`,
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
    env!("CARGO_BIN_EXE_ben-process")
}

// Six-node ring:
//     0 - 1 - 2 - 3 - 4 - 5 - 0
// with "pop" = 10 * (idx + 1), "area" = idx + 1, and "region" = A/A/B/B/A/A.
// Edge "weight" varies per edge so we can exercise --edge-weight-key.
fn write_fixture_graph(path: &Path) {
    let graph_json = serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "pop": 10.0, "area": 1.0, "region": "A" },
            { "pop": 20.0, "area": 2.0, "region": "A" },
            { "pop": 30.0, "area": 3.0, "region": "B" },
            { "pop": 40.0, "area": 4.0, "region": "B" },
            { "pop": 50.0, "area": 5.0, "region": "A" },
            { "pop": 60.0, "area": 6.0, "region": "A" },
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

// Four-node path:
//     0 - 1 - 2 - 3
// with:
//   area = 1 for every node
//   perim = 4 for every node
//   boundary_perim = [3, 2, 2, 3]
//   shared_perim = 1 for every edge
//
// This gives a simple Polsby-Popper fixture where total node perimeter can be
// supplied directly (`perim`) or derived exactly from
// `boundary_perim + shared_perim`.
fn write_polsby_fixture_graph(path: &Path) {
    let graph_json = serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "area": 1.0, "perim": 4.0, "boundary_perim": 3.0 },
            { "area": 1.0, "perim": 4.0, "boundary_perim": 2.0 },
            { "area": 1.0, "perim": 4.0, "boundary_perim": 2.0 },
            { "area": 1.0, "perim": 4.0, "boundary_perim": 3.0 }
        ],
        "adjacency": [
            [{ "id": 1, "shared_perim": 1.0 }],
            [{ "id": 0, "shared_perim": 1.0 }, { "id": 2, "shared_perim": 1.0 }],
            [{ "id": 1, "shared_perim": 1.0 }, { "id": 3, "shared_perim": 1.0 }],
            [{ "id": 2, "shared_perim": 1.0 }]
        ]
    });
    let mut f = File::create(path).unwrap();
    f.write_all(graph_json.to_string().as_bytes()).unwrap();
}

// Same four-node path as `write_polsby_fixture_graph`, but with GerryChain-like
// partial boundary perimeter data: only boundary nodes carry `boundary_perim`.
// The middle nodes omit the key entirely.
fn write_polsby_partial_boundary_graph(path: &Path) {
    let graph_json = serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "area": 1.0, "boundary_perim": 3.0 },
            { "area": 1.0 },
            { "area": 1.0 },
            { "area": 1.0, "boundary_perim": 3.0 }
        ],
        "adjacency": [
            [{ "id": 1, "shared_perim": 1.0 }],
            [{ "id": 0, "shared_perim": 1.0 }, { "id": 2, "shared_perim": 1.0 }],
            [{ "id": 1, "shared_perim": 1.0 }, { "id": 3, "shared_perim": 1.0 }],
            [{ "id": 2, "shared_perim": 1.0 }]
        ]
    });
    let mut f = File::create(path).unwrap();
    f.write_all(graph_json.to_string().as_bytes()).unwrap();
}

// Four-node path with one internal shared-perimeter edge omitted entirely.
// `frcw` treats missing shared_perim as 0.0 during derivation.
fn write_polsby_missing_shared_perim_graph(path: &Path) {
    let graph_json = serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "area": 1.0, "boundary_perim": 3.0 },
            { "area": 1.0, "boundary_perim": 2.0 },
            { "area": 1.0, "boundary_perim": 2.0 },
            { "area": 1.0, "boundary_perim": 3.0 }
        ],
        "adjacency": [
            [{ "id": 1, "shared_perim": 1.0 }],
            [{ "id": 0, "shared_perim": 1.0 }, { "id": 2 }],
            [{ "id": 1 }, { "id": 3, "shared_perim": 1.0 }],
            [{ "id": 2, "shared_perim": 1.0 }]
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
    Fixture {
        _tmp: tmp,
        dir,
        graph,
        ben,
    }
}

fn polsby_fixture(plans: &[Vec<u16>]) -> Fixture {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_polsby_fixture_graph(&graph);
    write_fixture_ben(&ben, plans);
    Fixture {
        _tmp: tmp,
        dir,
        graph,
        ben,
    }
}

fn polsby_partial_boundary_fixture(plans: &[Vec<u16>]) -> Fixture {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_polsby_partial_boundary_graph(&graph);
    write_fixture_ben(&ben, plans);
    Fixture {
        _tmp: tmp,
        dir,
        graph,
        ben,
    }
}

fn polsby_missing_shared_perim_fixture(plans: &[Vec<u16>]) -> Fixture {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_polsby_missing_shared_perim_graph(&graph);
    write_fixture_ben(&ben, plans);
    Fixture {
        _tmp: tmp,
        dir,
        graph,
        ben,
    }
}

fn run(args: &[&str]) {
    let status = Command::new(bin())
        .args(args)
        .status()
        .expect("failed to spawn ben-process");
    assert!(status.success(), "ben-process exited non-zero");
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

fn assert_f64_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual={actual:?} expected={expected:?}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "value mismatch at index {i}: actual={a} expected={e}"
        );
    }
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
    assert_eq!(u32_col(&df, "accepted_count"), vec![1, 2, 3]);
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

#[test]
fn tally_keys_pop_per_district() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode",
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_tallies").join("pop_tally_plans.parquet"));
    assert_eq!(f64_col(&df, "district_1"), vec![60.0, 90.0, 140.0]);
    assert_eq!(f64_col(&df, "district_2"), vec![150.0, 120.0, 70.0]);
    assert_eq!(u64_col(&df, "step"), vec![1, 2, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![1, 1, 1]);
    assert_eq!(u32_col(&df, "accepted_count"), vec![1, 2, 3]);
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
        "--mode",
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
        "--no-progress",
    ]);

    let pop_df = read_parquet(&f.dir.join("plans_tallies").join("pop_tally_plans.parquet"));
    let area_df = read_parquet(&f.dir.join("plans_tallies").join("area_tally_plans.parquet"));

    assert_eq!(f64_col(&pop_df, "district_1"), vec![60.0, 90.0, 140.0]);
    assert_eq!(f64_col(&pop_df, "district_2"), vec![150.0, 120.0, 70.0]);
    assert_eq!(f64_col(&area_df, "district_1"), vec![6.0, 9.0, 14.0]);
    assert_eq!(f64_col(&area_df, "district_2"), vec![15.0, 12.0, 7.0]);
    assert_eq!(u64_col(&area_df, "step"), vec![1, 2, 3]);
    assert_eq!(u32_col(&area_df, "accepted_count"), vec![1, 2, 3]);
}

#[test]
fn tally_keys_output_dir_nests_files_under_graph_stem_directory() {
    let f = fixture(&tri_plans());
    let output_dir = f.dir.join("custom_out");
    run(&[
        "--mode",
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        output_dir.to_str().unwrap(),
        "--keys",
        "pop",
        "--no-progress",
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

#[test]
fn tally_keys_fails_when_later_frames_introduce_unseen_district_ids() {
    let plans = vec![vec![1u16, 1, 1, 1, 1, 1], vec![1u16, 2, 1, 2, 1, 2]];
    let f = fixture(&plans);
    let output = Command::new(bin())
        .args([
            "--mode",
            "tally-keys",
            "--graph-file",
            f.graph.to_str().unwrap(),
            "--ben-file",
            f.ben.to_str().unwrap(),
            "--output-dir",
            f.dir.to_str().unwrap(),
            "--keys",
            "pop",
            "--no-progress",
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

#[test]
fn polsby_popper_with_explicit_perimeter_key() {
    let plans = vec![vec![1u16, 1, 2, 2], vec![1u16, 2, 2, 2]];
    let f = polsby_fixture(&plans);
    run(&[
        "--mode",
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
        "--no-progress",
    ]);

    let df = read_parquet(&f.dir.join("plans_polsby_popper.parquet"));
    let expected_d1 = [2.0 * std::f64::consts::PI / 9.0, std::f64::consts::PI / 4.0];
    let expected_d2 = [
        2.0 * std::f64::consts::PI / 9.0,
        3.0 * std::f64::consts::PI / 16.0,
    ];
    assert_f64_vec_close(&f64_col(&df, "district_1"), &expected_d1);
    assert_f64_vec_close(&f64_col(&df, "district_2"), &expected_d2);
    assert_eq!(u64_col(&df, "step"), vec![1, 2]);
    assert_eq!(u32_col(&df, "n_reps"), vec![1, 1]);
    assert_eq!(u32_col(&df, "accepted_count"), vec![1, 2]);
}

#[test]
fn polsby_popper_with_boundary_and_shared_perimeter_matches_direct_perimeter() {
    let plans = vec![vec![1u16, 1, 2, 2], vec![1u16, 2, 2, 2]];
    let f = polsby_fixture(&plans);
    run(&[
        "--mode",
        "polsby-popper",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--area-key",
        "area",
        "--boundary-perim-key",
        "boundary_perim",
        "--shared-perim-key",
        "shared_perim",
        "--no-progress",
    ]);

    let df = read_parquet(&f.dir.join("plans_polsby_popper.parquet"));
    let expected_d1 = [2.0 * std::f64::consts::PI / 9.0, std::f64::consts::PI / 4.0];
    let expected_d2 = [
        2.0 * std::f64::consts::PI / 9.0,
        3.0 * std::f64::consts::PI / 16.0,
    ];
    assert_f64_vec_close(&f64_col(&df, "district_1"), &expected_d1);
    assert_f64_vec_close(&f64_col(&df, "district_2"), &expected_d2);
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
fn polsby_popper_uses_gerrychain_default_geometry_keys() {
    let plans = vec![vec![1u16, 1, 2, 2]];
    let f = polsby_fixture(&plans);
    run(&[
        "--mode",
        "polsby-popper",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    let df = read_parquet(&f.dir.join("plans_polsby_popper.parquet"));
    let expected = [2.0 * std::f64::consts::PI / 9.0];
    assert_f64_vec_close(&f64_col(&df, "district_1"), &expected);
    assert_f64_vec_close(&f64_col(&df, "district_2"), &expected);
}

#[test]
fn polsby_popper_treats_missing_boundary_perimeter_as_zero() {
    let plans = vec![vec![1u16, 1, 2, 2]];
    let f = polsby_partial_boundary_fixture(&plans);
    run(&[
        "--mode",
        "polsby-popper",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    let df = read_parquet(&f.dir.join("plans_polsby_popper.parquet"));
    let expected = [std::f64::consts::PI / 2.0];
    assert_f64_vec_close(&f64_col(&df, "district_1"), &expected);
    assert_f64_vec_close(&f64_col(&df, "district_2"), &expected);
}

#[test]
fn polsby_popper_treats_missing_shared_perimeter_as_zero() {
    let plans = vec![vec![1u16, 1, 2, 2]];
    let f = polsby_missing_shared_perim_fixture(&plans);
    run(&[
        "--mode",
        "polsby-popper",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    let df = read_parquet(&f.dir.join("plans_polsby_popper.parquet"));
    let expected = [8.0 * std::f64::consts::PI / 25.0];
    assert_f64_vec_close(&f64_col(&df, "district_1"), &expected);
    assert_f64_vec_close(&f64_col(&df, "district_2"), &expected);
}

#[test]
fn polsby_popper_uses_default_area_and_shared_keys_with_explicit_boundary_perimeter_key() {
    let plans = vec![vec![1u16, 1, 2, 2]];
    let f = polsby_fixture(&plans);
    run(&[
        "--mode",
        "polsby-popper",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--boundary-perim-key",
        "boundary_perim",
        "--no-progress",
    ]);

    let df = read_parquet(&f.dir.join("plans_polsby_popper.parquet"));
    let expected = [2.0 * std::f64::consts::PI / 9.0];
    assert_f64_vec_close(&f64_col(&df, "district_1"), &expected);
    assert_f64_vec_close(&f64_col(&df, "district_2"), &expected);
}

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

#[test]
fn changed_assignments_single_plan_smoke() {
    let plans = vec![vec![1u16, 1, 1, 2, 2, 2]];
    let f = fixture(&plans);
    run(&[
        "--mode",
        "changed-assignments",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let body =
        std::fs::read_to_string(f.dir.join("plans_accept_1_changed_assignments.txt")).unwrap();
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
        "--mode",
        "changed-assignments",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let body =
        std::fs::read_to_string(f.dir.join("plans_accept_3_changed_assignments.txt")).unwrap();
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
        "--mode",
        "changed-assignments",
        "--ben-file",
        ben.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
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

/// Normalize divides each count by `line_count - 1` = 2.
#[test]
fn changed_assignments_tri_plans_normalized() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode",
        "changed-assignments",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--normalize",
        "--no-progress",
    ]);
    let body =
        std::fs::read_to_string(f.dir.join("plans_accept_3_changed_assignments.txt")).unwrap();
    assert_eq!(body, "[0.0, 1.0, 0.5, 0.0, 0.5, 0.5]\nTotal Accepted: 3");
}

#[test]
fn changed_assignments_respects_max_accepted() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode",
        "changed-assignments",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--max-accepted",
        "2",
        "--no-progress",
    ]);
    let body =
        std::fs::read_to_string(f.dir.join("plans_accept_2_changed_assignments.txt")).unwrap();
    assert_eq!(body, "[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]\nTotal Accepted: 2");
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
        "--mode",
        "extract-unique-plans",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    let out = f.dir.join("plans_unique.jsonl.ben");
    let decoder = BenDecoder::new(File::open(&out).unwrap()).unwrap();
    let extracted: Vec<Vec<u16>> = decoder.map(|r| r.unwrap().0).collect();

    assert_eq!(
        extracted,
        vec![
            vec![1u16, 1, 1, 2, 2, 2], // P_A first occurrence (original labels)
            vec![1u16, 1, 2, 2, 1, 1], // P_B first occurrence
            vec![1u16, 2, 1, 2, 1, 2], // P_C first occurrence
        ]
    );
}

/// `--high-compression` switches the parquet writer from Snappy to Brotli.
/// None of the other snapshot tests exercise this branch, so a regression in
/// the Brotli writer setup wouldn't be caught. Run cut-edges with
/// --high-compression and verify the output is still a valid parquet file
/// with the expected values; the polars reader is compression-agnostic.
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

/// `extract-unique-plans` opens the input via `BenDecoder::new` after a
/// `count_frames` pass, both of which return errors that propagate via `?`.
/// Feed it a non-BEN file and assert the binary exits non-zero rather than
/// silently producing an empty/corrupt output. This guards the error path
/// in src/metrics/extract_unique_plans.rs and src/pipeline.rs (count_frames).
#[test]
fn extract_unique_plans_fails_on_corrupted_ben_input() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bogus = dir.join("not_a_real.jsonl.ben");
    // Bytes that are definitely not a valid BEN stream — neither the magic
    // header nor any decodable frames.
    std::fs::write(&bogus, b"this is not a BEN file").unwrap();

    let output = Command::new(bin())
        .args([
            "--mode",
            "extract-unique-plans",
            "--ben-file",
            bogus.to_str().unwrap(),
            "--output-dir",
            dir.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .expect("failed to spawn ben-process");

    assert!(
        !output.status.success(),
        "extract-unique-plans should fail on a corrupted BEN input; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

/// When both `--perim-key` and `--boundary-perim-key` are passed, `perim-key`
/// must take precedence (main.rs uses the direct perimeter and ignores the
/// boundary derivation in that case). Build a graph where `boundary_perim` is
/// deliberately wrong (would derive `total_perim = 1002` per node and push the
/// score toward zero); the only way to get the canonical
/// `2 * PI / 9` score is for the perim-key path to win.
#[test]
fn polsby_popper_perim_key_wins_when_both_keys_are_passed() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");

    // Same area / shared_perim / perim as the standard polsby fixture, but
    // boundary_perim is set to 999.0 — derivation would yield ~1001 per node
    // and a near-zero score. perim=4 yields 2*PI/9 for plan [1,1,2,2].
    let graph_json = serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "area": 1.0, "perim": 4.0, "boundary_perim": 999.0 },
            { "area": 1.0, "perim": 4.0, "boundary_perim": 999.0 },
            { "area": 1.0, "perim": 4.0, "boundary_perim": 999.0 },
            { "area": 1.0, "perim": 4.0, "boundary_perim": 999.0 }
        ],
        "adjacency": [
            [{ "id": 1, "shared_perim": 1.0 }],
            [{ "id": 0, "shared_perim": 1.0 }, { "id": 2, "shared_perim": 1.0 }],
            [{ "id": 1, "shared_perim": 1.0 }, { "id": 3, "shared_perim": 1.0 }],
            [{ "id": 2, "shared_perim": 1.0 }]
        ]
    });
    std::fs::write(&graph, graph_json.to_string()).unwrap();
    write_fixture_ben(&ben, &[vec![1u16, 1, 2, 2]]);

    run(&[
        "--mode",
        "polsby-popper",
        "--graph-file",
        graph.to_str().unwrap(),
        "--ben-file",
        ben.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "--area-key",
        "area",
        "--perim-key",
        "perim",
        "--boundary-perim-key",
        "boundary_perim",
        "--shared-perim-key",
        "shared_perim",
        "--no-progress",
    ]);

    let df = read_parquet(&dir.join("plans_polsby_popper.parquet"));
    let expected = [2.0 * std::f64::consts::PI / 9.0];
    assert_f64_vec_close(&f64_col(&df, "district_1"), &expected);
    assert_f64_vec_close(&f64_col(&df, "district_2"), &expected);
}

/// `--randomize-reassignments` uses a thread-local OS-seeded RNG, so we can't
/// pin exact dif counts. This is a smoke test: confirm the binary completes
/// successfully, the output file uses the expected frame count, and the
/// per-unit values are within the valid range. With two label-permuted frames,
/// each per-unit count is either 0 (RNG fired and labels were swapped before
/// counting) or 1 (RNG didn't fire; pure diff).
#[test]
fn changed_assignments_with_randomize_reassignments_runs_and_writes_valid_output() {
    let plans = vec![vec![1u16, 1, 2, 2], vec![2u16, 2, 1, 1]];
    let f = fixture(&plans);
    run(&[
        "--mode",
        "changed-assignments",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--randomize-reassignments",
        "--no-progress",
    ]);

    let body =
        std::fs::read_to_string(f.dir.join("plans_accept_2_changed_assignments.txt")).unwrap();
    let (counts_line, total_line) = body.split_once('\n').expect("output should have two lines");
    assert_eq!(total_line, "Total Accepted: 2");

    let parsed: Vec<f64> = counts_line
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(", ")
        .map(|s| s.parse::<f64>().unwrap())
        .collect();
    assert_eq!(parsed.len(), 4);
    for v in parsed {
        assert!(
            v == 0.0 || v == 1.0,
            "per-unit count under --randomize-reassignments must be 0 or 1, got {v}"
        );
    }
}

/// `--output-dir` pointing at an existing *file* (not a directory) cannot host
/// the output parquet. `File::create` will fail; the binary must surface a
/// non-zero exit rather than silently producing nothing.
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

/// `polsby-popper` has an explicit "no rows seen" branch that builds an empty
/// schema-less DataFrame and finishes the parquet writer without ever calling
/// `sorted_district_ids`. Drive that branch with a BEN file containing zero
/// frames and verify the binary exits cleanly and produces a readable parquet.
#[test]
fn polsby_popper_handles_empty_ben_input() {
    let plans: Vec<Vec<u16>> = vec![];
    let f = polsby_fixture(&plans);
    run(&[
        "--mode",
        "polsby-popper",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let df = read_parquet(&f.dir.join("plans_polsby_popper.parquet"));
    assert_eq!(df.height(), 0);
    // The empty branch builds the schema with no district columns.
    assert_eq!(
        df.get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>(),
        vec![
            "step".to_string(),
            "n_reps".to_string(),
            "accepted_count".to_string()
        ]
    );
}

/// `cut-edges` and `tally-keys` both go through `run_pipeline`; on a zero-frame
/// BEN they must finish the parquet writer without ever invoking the per-row
/// callback. Smoke-test both modes — the existence of a readable empty parquet
/// is enough to catch a regression that panics on the empty-iterator path.
#[test]
fn cut_edges_and_tally_keys_handle_empty_ben_input() {
    let plans: Vec<Vec<u16>> = vec![];
    let f = fixture(&plans);

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
    let cut_df = read_parquet(&f.dir.join("plans_cut_edges.parquet"));
    assert_eq!(cut_df.height(), 0);

    run(&[
        "--mode",
        "tally-keys",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--keys",
        "pop",
        "--no-progress",
    ]);
    let tally_df = read_parquet(&f.dir.join("plans_tallies").join("pop_tally_plans.parquet"));
    assert_eq!(tally_df.height(), 0);
}

/// All five frames are byte-identical → exactly one canonical partition.
#[test]
fn unique_plans_reports_one_when_every_frame_is_identical() {
    let plans: Vec<Vec<u16>> = vec![vec![1, 1, 1, 2, 2, 2]; 5];
    let f = fixture(&plans);
    run(&[
        "--mode",
        "unique-plans",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let contents = std::fs::read_to_string(f.dir.join("plans_unique_plans.txt")).unwrap();
    assert_eq!(contents, "unique_plans: 1\ntotal_accepted_frames: 5\n");
}

/// Five frames, every one a distinct partition (not just a label permutation
/// of any other) → unique count equals frame count.
#[test]
fn unique_plans_reports_n_when_every_frame_is_distinct() {
    let plans: Vec<Vec<u16>> = vec![
        vec![1, 1, 1, 2, 2, 2], // d1 = first three
        vec![1, 1, 2, 2, 1, 1], // d2 = middle two
        vec![1, 2, 1, 2, 1, 2], // alternating
        vec![1, 1, 1, 1, 2, 2], // d2 = last two
        vec![1, 2, 2, 1, 1, 2], // mixed
    ];
    let f = fixture(&plans);
    run(&[
        "--mode",
        "unique-plans",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    let contents = std::fs::read_to_string(f.dir.join("plans_unique_plans.txt")).unwrap();
    assert_eq!(contents, "unique_plans: 5\ntotal_accepted_frames: 5\n");
}

#[test]
fn unique_plans_writes_distinct_partition_count_and_total_frames() {
    // Same fixture as extract_unique_plans: 3 distinct partitions among 5 frames
    // (P_A, P_B, P_A-relabeled, P_B-duplicate, P_C).
    let plans: Vec<Vec<u16>> = vec![
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 1, 2, 2, 1, 1],
        vec![2, 2, 2, 1, 1, 1],
        vec![1, 1, 2, 2, 1, 1],
        vec![1, 2, 1, 2, 1, 2],
    ];
    let f = fixture(&plans);
    run(&[
        "--mode",
        "unique-plans",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);

    let out = f.dir.join("plans_unique_plans.txt");
    let contents = std::fs::read_to_string(&out).unwrap();
    assert_eq!(contents, "unique_plans: 3\ntotal_accepted_frames: 5\n");
}
