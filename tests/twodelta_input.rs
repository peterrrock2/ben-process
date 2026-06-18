//! TwoDelta `.ben` input works for free after the 1.0 migration: the frame reader replays delta
//! frames against its running previous-assignment on the serial pop thread and hands the pipeline
//! self-contained frames, so every graph mode sees the same assignments as a Standard encoding.

#[path = "common/mod.rs"]
mod common;

use common::*;

/// Run `cut-edges` on a `TwoDelta`-encoded copy of `tri_plans` over the 6-node ring. The result
/// must equal the Standard encoding's hand-computed unweighted cut counts (2, 6, 2).
#[test]
fn cut_edges_on_twodelta_matches_standard() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    write_fixture_graph(&graph);
    let ben = dir.join("plans.jsonl.ben");
    write_fixture_ben_variant(&ben, &tri_plans(), BenVariant::TwoDelta);

    run(&[
        "--mode",
        "cut-edges",
        "--graph-file",
        graph.to_str().unwrap(),
        "--ben-file",
        ben.to_str().unwrap(),
        "--output-dir",
        dir.to_str().unwrap(),
        "--no-progress",
    ]);

    let df = read_parquet(&dir.join("plans_cut_edges.parquet"));
    assert_f64_vec_close(&f64_col(&df, "cut_edges"), &[2.0, 6.0, 2.0]);
}
