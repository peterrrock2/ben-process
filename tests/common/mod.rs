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

pub(crate) use ben::io::bundle::format::{AssignmentFormat, ASSET_TYPE_GRAPH};
pub(crate) use ben::io::bundle::BendlWriter;
pub(crate) use ben::io::reader::BenStreamReader;
pub(crate) use ben::io::writer::{BenStreamWriter, XzEncodeOptions};
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
pub(crate) fn fixture_graph_bytes() -> Vec<u8> {
    serde_json::json!({
        "directed": false,
        "multigraph": false,
        "graph": [],
        "nodes": [
            { "id": 0, "pop": 10.0, "area": 1.0, "region": "A" },
            { "id": 1, "pop": 20.0, "area": 2.0, "region": "A" },
            { "id": 2, "pop": 30.0, "area": 3.0, "region": "B" },
            { "id": 3, "pop": 40.0, "area": 4.0, "region": "B" },
            { "id": 4, "pop": 50.0, "area": 5.0, "region": "A" },
            { "id": 5, "pop": 60.0, "area": 6.0, "region": "A" },
        ],
        "adjacency": [
            [ { "id": 1, "weight": 2.0 }, { "id": 5, "weight": 3.0 } ],
            [ { "id": 0, "weight": 2.0 }, { "id": 2, "weight": 1.0 } ],
            [ { "id": 1, "weight": 1.0 }, { "id": 3, "weight": 5.0 } ],
            [ { "id": 2, "weight": 5.0 }, { "id": 4, "weight": 4.0 } ],
            [ { "id": 3, "weight": 4.0 }, { "id": 5, "weight": 6.0 } ],
            [ { "id": 0, "weight": 3.0 }, { "id": 4, "weight": 6.0 } ],
        ]
    })
    .to_string()
    .into_bytes()
}

pub(crate) fn write_fixture_graph(path: &Path) {
    File::create(path)
        .unwrap()
        .write_all(&fixture_graph_bytes())
        .unwrap();
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
    let mut enc = BenStreamWriter::for_ben(f, BenVariant::Standard).unwrap();
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
    let mut enc = BenStreamWriter::for_ben(f, BenVariant::MkvChain).unwrap();
    for p in plans {
        enc.write_assignment(p.clone()).unwrap();
    }
    enc.finish().unwrap();
}

/// Encode `plans` with an explicit `BenVariant` (used to mint a `TwoDelta` fixture).
pub(crate) fn write_fixture_ben_variant(path: &Path, plans: &[Vec<u16>], variant: BenVariant) {
    let f = File::create(path).unwrap();
    let mut enc = BenStreamWriter::for_ben(f, variant).unwrap();
    for p in plans {
        enc.write_assignment(p.clone()).unwrap();
    }
    enc.finish().unwrap();
}

/// Write a `.bendl` bundle: an optional `graph.json` asset plus an assignment stream (BEN or XBEN),
/// finalized with the header sample count set to `sample_count`.
fn write_bendl(
    path: &Path,
    graph: Option<&[u8]>,
    plans: &[Vec<u16>],
    xben: bool,
    sample_count: i64,
) {
    let format = if xben {
        AssignmentFormat::Xben
    } else {
        AssignmentFormat::Ben
    };
    let mut writer = BendlWriter::new(File::create(path).unwrap(), format).unwrap();
    if let Some(graph) = graph {
        writer
            .add_json_asset(ASSET_TYPE_GRAPH, "graph.json", graph)
            .unwrap();
    }
    let mut session = writer.into_stream_session().unwrap();
    if xben {
        let mut enc =
            BenStreamWriter::for_xben(&mut session, BenVariant::Standard, XzEncodeOptions::new())
                .unwrap();
        for p in plans {
            enc.write_assignment(p.clone()).unwrap();
        }
        enc.finish().unwrap();
    } else {
        let mut enc = BenStreamWriter::for_ben(&mut session, BenVariant::Standard).unwrap();
        for p in plans {
            enc.write_assignment(p.clone()).unwrap();
        }
        enc.finish().unwrap();
    }
    let writer = session.finish_into_writer(sample_count);
    writer.finish().unwrap();
}

/// A finalized `.bendl` with a BEN stream and (optionally) an embedded `graph.json`.
pub(crate) fn write_bendl_ben(path: &Path, graph: Option<&[u8]>, plans: &[Vec<u16>]) {
    write_bendl(path, graph, plans, false, plans.len() as i64);
}

/// A finalized `.bendl` with an XBEN (xz-compressed) stream.
pub(crate) fn write_bendl_xben(path: &Path, graph: Option<&[u8]>, plans: &[Vec<u16>]) {
    write_bendl(path, graph, plans, true, plans.len() as i64);
}

/// A finalized BEN `.bendl` with a deliberately chosen header sample count, for the
/// resolution-time count-validation test.
pub(crate) fn write_bendl_ben_sample_count(
    path: &Path,
    graph: Option<&[u8]>,
    plans: &[Vec<u16>],
    sample_count: i64,
) {
    write_bendl(path, graph, plans, false, sample_count);
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

/// Like [`fixture`] but encodes the plans as MkvChain BEN, so consecutive identical assignments
/// coalesce into one frame with `count > 1`. Used to verify the `step`/`n_reps` accounting.
pub(crate) fn fixture_mkv(plans: &[Vec<u16>]) -> Fixture {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_fixture_graph(&graph);
    write_fixture_ben_mkv(&ben, plans);
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

/// MkvChain counterpart of [`polsby_fixture`].
pub(crate) fn polsby_fixture_mkv(plans: &[Vec<u16>]) -> Fixture {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");
    write_polsby_fixture_graph(&graph);
    write_fixture_ben_mkv(&ben, plans);
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

/// Like [`run`], but returns captured stderr so tests can assert on warnings/log output of a
/// successful run.
pub(crate) fn run_success_capture_stderr(args: &[&str]) -> String {
    let output = Command::new(bin())
        .args(args)
        .output()
        .expect("failed to spawn ben-process");
    assert!(
        output.status.success(),
        "ben-process exited non-zero; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8_lossy(&output.stderr).into_owned()
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
