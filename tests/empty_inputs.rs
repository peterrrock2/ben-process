#[path = "common/mod.rs"]
mod common;

use common::*;

/// `cut-edges` and `tally-keys` both go through `run_pipeline`; on a zero-frame BEN they must
/// finish the parquet writer without ever invoking the per-row callback. Smoke-test both modes —
/// the existence of a readable empty parquet is enough to catch a regression that panics on the
/// empty-iterator path.
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
