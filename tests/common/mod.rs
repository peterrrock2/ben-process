//! End-to-end regression tests for ben-process.
//!
//! Each test builds a tiny 6-node ring fixture + a handful of assignment vectors, invokes the
//! compiled `ben-process` binary via `env!("CARGO_BIN_EXE_ben-process")`, and asserts the produced
//! Parquet / text output against manually-computed expected values. The fixture is intentionally
//! small enough that a reader can verify every expected value on paper.
#![allow(dead_code, unused_imports)]

pub(crate) use std::fs::File;
pub(crate) use std::io::Write;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::process::Command;

pub(crate) use ben::decode::BenDecoder;
pub(crate) use ben::encode::BenEncoder;
pub(crate) use ben::BenVariant;
use polars::prelude::*;
pub(crate) use tempfile::{tempdir, TempDir};

pub(crate) fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_ben-process")
}

// Six-node ring:
//     0 - 1 - 2 - 3 - 4 - 5 - 0
// with "pop" = 10 * (idx + 1), "area" = idx + 1, and "region" = A/A/B/B/A/A. Edge "weight" varies
// per edge so we can exercise --edge-weight-key.
pub(crate) fn write_fixture_graph(path: &Path) {
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
// This gives a simple Polsby-Popper fixture where total node perimeter can be supplied directly
// (`perim`) or derived exactly from `boundary_perim + shared_perim`.
pub(crate) fn write_polsby_fixture_graph(path: &Path) {
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

// Same four-node path as `write_polsby_fixture_graph`, but with GerryChain-like partial boundary
// perimeter data: only boundary nodes carry `boundary_perim`. The middle nodes omit the key
// entirely.
pub(crate) fn write_polsby_partial_boundary_graph(path: &Path) {
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

// Four-node path with one internal shared-perimeter edge omitted entirely. `frcw` treats missing
// shared_perim as 0.0 during derivation.
pub(crate) fn write_polsby_missing_shared_perim_graph(path: &Path) {
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

pub(crate) fn write_fixture_ben(path: &Path, plans: &[Vec<u16>]) {
    let f = File::create(path).unwrap();
    let mut enc = BenEncoder::new(f, BenVariant::Standard);
    for p in plans {
        enc.write_assignment(p.clone()).unwrap();
    }
    enc.finish().unwrap();
}

/// Encode with `BenVariant::MkvChain` so consecutive identical assignments collapse into a single
/// frame with `count > 1`. Used by the MkvChain regression test for changed-assignments frame
/// counting.
pub(crate) fn write_fixture_ben_mkv(path: &Path, plans: &[Vec<u16>]) {
    let f = File::create(path).unwrap();
    let mut enc = BenEncoder::new(f, BenVariant::MkvChain);
    for p in plans {
        enc.write_assignment(p.clone()).unwrap();
    }
    enc.finish().unwrap();
}

pub(crate) struct Fixture {
    pub(crate) _tmp: TempDir, // keeps the temp dir alive for the duration of the test
    pub(crate) dir: PathBuf,
    pub(crate) graph: PathBuf,
    pub(crate) ben: PathBuf,
}

pub(crate) fn fixture(plans: &[Vec<u16>]) -> Fixture {
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

pub(crate) fn polsby_fixture(plans: &[Vec<u16>]) -> Fixture {
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

pub(crate) fn polsby_partial_boundary_fixture(plans: &[Vec<u16>]) -> Fixture {
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

pub(crate) fn polsby_missing_shared_perim_fixture(plans: &[Vec<u16>]) -> Fixture {
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

pub(crate) fn run(args: &[&str]) {
    let status = Command::new(bin())
        .args(args)
        .status()
        .expect("failed to spawn ben-process");
    assert!(status.success(), "ben-process exited non-zero");
}

pub(crate) fn run_failure(args: &[&str]) -> String {
    let output = Command::new(bin())
        .args(args)
        .output()
        .expect("failed to spawn ben-process");
    assert!(
        !output.status.success(),
        "ben-process should have failed; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
}

pub(crate) fn read_parquet(path: &Path) -> DataFrame {
    ParquetReader::new(&mut File::open(path).unwrap())
        .finish()
        .unwrap()
}

pub(crate) fn f64_col(df: &DataFrame, name: &str) -> Vec<f64> {
    df.column(name)
        .unwrap()
        .f64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

pub(crate) fn u32_col(df: &DataFrame, name: &str) -> Vec<u32> {
    df.column(name)
        .unwrap()
        .u32()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

pub(crate) fn u64_col(df: &DataFrame, name: &str) -> Vec<u64> {
    df.column(name)
        .unwrap()
        .u64()
        .unwrap()
        .into_no_null_iter()
        .collect()
}

pub(crate) fn str_col(df: &DataFrame, name: &str) -> Vec<String> {
    df.column(name)
        .unwrap()
        .str()
        .unwrap()
        .into_no_null_iter()
        .map(|s| s.to_string())
        .collect()
}

pub(crate) fn assert_f64_vec_close(actual: &[f64], expected: &[f64]) {
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
pub(crate) fn tri_plans() -> Vec<Vec<u16>> {
    vec![
        vec![1, 1, 1, 2, 2, 2],
        vec![1, 2, 1, 2, 1, 2],
        vec![1, 1, 2, 2, 1, 1],
    ]
}
