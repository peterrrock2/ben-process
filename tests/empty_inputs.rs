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

/// `region-*` also goes through `run_pipeline`; a zero-frame BEN must finish the keyed writer and
/// leave a readable, empty parquet.
#[test]
fn region_splits_handles_empty_ben_input() {
    let plans: Vec<Vec<u16>> = vec![];
    let f = fixture(&plans);
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
    assert_eq!(df.height(), 0);
}

/// `changed-assignments` has nothing to tally on a zero-frame BEN; it must fail fast with a clear
/// message rather than write a meaningless empty file.
#[test]
fn changed_assignments_fails_on_empty_ben_input() {
    let plans: Vec<Vec<u16>> = vec![];
    let f = fixture(&plans);
    let stderr = run_failure(&[
        "--mode",
        "changed-assignments",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ]);
    assert!(
        stderr.contains("No data found"),
        "changed-assignments should report no data on empty input, got: {stderr}"
    );
}

/// `unique-plans` on a zero-frame BEN should report zero of both counts.
#[test]
fn unique_plans_handles_empty_ben_input() {
    let plans: Vec<Vec<u16>> = vec![];
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
    let df = read_parquet(&f.dir.join("plans_unique_plans.parquet"));
    assert_eq!(u64_col(&df, "unique_plans"), vec![0]);
    assert_eq!(u64_col(&df, "total_accepted_frames"), vec![0]);
}

/// `extract-unique-plans` on a zero-frame BEN should produce a valid, empty Standard BEN that
/// decodes to no plans.
#[test]
fn extract_unique_plans_handles_empty_ben_input() {
    let plans: Vec<Vec<u16>> = vec![];
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
    assert!(
        extracted.is_empty(),
        "expected no extracted plans, got {extracted:?}"
    );
}
