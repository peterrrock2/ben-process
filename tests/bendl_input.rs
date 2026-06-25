//! `.bendl` bundle input: the stream is read natively (no temp copy), the embedded `graph.json`
//! satisfies graph-driven modes without `--graph-file`, `--graph-file` overrides an embedded graph,
//! output names derive from the `.bendl` basename, and graph-free modes run straight off the
//! stream.

#[path = "common/mod.rs"]
mod common;

use ben::io::bundle::format::{
    write_header_with_tail, BendlHeader, HEADER_SIZE, HEADER_WITH_TAIL_SIZE,
};
use common::*;

fn corrupt_stream_len_past_eof(path: &Path) {
    let mut bytes = std::fs::read(path).unwrap();
    let header_bytes: [u8; HEADER_SIZE] = bytes[..HEADER_SIZE].try_into().unwrap();
    let mut header = BendlHeader::from_bytes(&header_bytes).unwrap();
    header.stream_len = bytes.len() as u64 + 1;

    let mut patched_header = Vec::new();
    write_header_with_tail(&mut patched_header, &header).unwrap();
    assert_eq!(patched_header.len(), HEADER_WITH_TAIL_SIZE);
    bytes[..HEADER_WITH_TAIL_SIZE].copy_from_slice(&patched_header);
    std::fs::write(path, bytes).unwrap();
}

/// A `.bendl` carrying `graph.json` + a BEN stream: `cut-edges` with **no** `--graph-file` uses the
/// embedded graph and produces the ring's hand-computed unweighted cut counts (2, 6, 2). The output
/// name derives from the `.bendl` basename (`plans.bendl` -> `plans_cut_edges.parquet`).
#[test]
fn bendl_uses_embedded_graph_without_graph_file() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bendl = dir.join("plans.bendl");
    write_bendl_ben(&bendl, Some(&fixture_graph_bytes()), &tri_plans());

    run(&[
        "cut-edges",
        "--ben-file",
        bendl.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);

    let df = read_parquet(&dir.join("plans_cut_edges.parquet"));
    assert_f64_vec_close(&f64_col(&df, "cut_edges"), &[2.0, 6.0, 2.0]);
}

/// `--graph-file` wins over an embedded graph. The bundle embeds deliberately invalid graph bytes
/// (CRC-valid but not JSON); the run still succeeds because the explicit `--graph-file` is used and
/// the embedded bytes are never parsed.
#[test]
fn bendl_graph_file_overrides_embedded() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bendl = dir.join("plans.bendl");
    write_bendl_ben(&bendl, Some(b"this is not valid graph json"), &tri_plans());
    let graph = dir.join("real_graph.json");
    write_fixture_graph(&graph);

    run(&[
        "cut-edges",
        "--graph-file",
        graph.to_str().unwrap(),
        "--ben-file",
        bendl.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);

    let df = read_parquet(&dir.join("plans_cut_edges.parquet"));
    assert_f64_vec_close(&f64_col(&df, "cut_edges"), &[2.0, 6.0, 2.0]);
}

/// Graph-driven mode + a bundle with no `graph.json` asset + no `--graph-file` is the standard
/// "graph file required" error.
#[test]
fn bendl_without_graph_and_no_graph_file_errors() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bendl = dir.join("plans.bendl");
    write_bendl_ben(&bendl, None, &tri_plans());

    let stderr = run_failure(&[
        "cut-edges",
        "--ben-file",
        bendl.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);
    assert!(
        stderr.contains("graph file required"),
        "expected the graph-required error, got: {stderr}"
    );
}

/// A graph-free mode (`unique-plans`) reads straight off the bundle stream with no graph at all.
/// The three `tri_plans` are distinct partitions, so the count is 3 of 3.
#[test]
fn bendl_unique_plans_runs_without_graph() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bendl = dir.join("plans.bendl");
    write_bendl_ben(&bendl, None, &tri_plans());

    run(&[
        "unique-plans",
        "--ben-file",
        bendl.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);

    let df = read_parquet(&dir.join("plans_unique_plans.parquet"));
    assert_eq!(u64_col(&df, "unique_plans"), vec![3]);
    assert_eq!(u64_col(&df, "total_accepted_frames"), vec![3]);
}

/// An XBEN (xz-compressed) bundle stream is read via `from_xben`; the embedded graph still drives
/// `cut-edges` to the same hand-computed counts.
#[test]
fn bendl_xben_stream_cut_edges() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bendl = dir.join("plans.bendl");
    write_bendl_xben(&bendl, Some(&fixture_graph_bytes()), &tri_plans());

    run(&[
        "cut-edges",
        "--ben-file",
        bendl.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);

    let df = read_parquet(&dir.join("plans_cut_edges.parquet"));
    assert_f64_vec_close(&f64_col(&df, "cut_edges"), &[2.0, 6.0, 2.0]);
}

/// A finalized bundle whose header `sample_count` is negative must fail at resolution (the
/// `usize::try_from` guard), before any frame is processed, for every mode.
#[test]
fn bendl_negative_sample_count_errors() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bendl = dir.join("plans.bendl");
    write_bendl_ben_sample_count(&bendl, None, &tri_plans(), -1);

    let stderr = run_failure(&[
        "unique-plans",
        "--ben-file",
        bendl.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);
    assert!(
        stderr.contains("sample_count") && stderr.contains("negative or out of range"),
        "expected the negative-sample-count error, got: {stderr}"
    );
}

/// A finalized bundle whose declared stream range extends past EOF must fail during input
/// resolution. Without this guard, a BEN reader could treat EOF at a frame boundary as a clean end
/// and silently decode only the prefix.
#[test]
fn bendl_stream_len_past_eof_errors_at_resolution() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bendl = dir.join("plans.bendl");
    write_bendl_ben(&bendl, None, &tri_plans());
    corrupt_stream_len_past_eof(&bendl);

    let stderr = run_failure(&[
        "unique-plans",
        "--ben-file",
        bendl.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "-q",
    ]);
    assert!(
        stderr.contains("bundle assignment stream range") && stderr.contains("exceeds file length"),
        "expected the past-EOF stream-range error, got: {stderr}"
    );
}
