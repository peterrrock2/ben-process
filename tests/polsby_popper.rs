#[path = "common/mod.rs"]
mod common;

use common::*;

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
    assert_eq!(u64_col(&df, "accepted_count"), vec![1, 2]);
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

/// When both `--perim-key` and `--boundary-perim-key` are passed, `perim-key` must take precedence
/// (main.rs uses the direct perimeter and ignores the boundary derivation in that case). Build a
/// graph where `boundary_perim` is deliberately wrong (would derive `total_perim = 1002` per node
/// and push the score toward zero); the only way to get the canonical `2 * PI / 9` score is for the
/// perim-key path to win.
#[test]
fn polsby_popper_perim_key_wins_when_both_keys_are_passed() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let graph = dir.join("graph.json");
    let ben = dir.join("plans.jsonl.ben");

    // Same area / shared_perim / perim as the standard polsby fixture, but boundary_perim is set to
    // 999.0 — derivation would yield ~1001 per node and a near-zero score. perim=4 yields 2*PI/9
    // for plan [1,1,2,2].
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

/// Same fixed-district-set invariant as tally-keys, exercised through the Polsby-Popper schema:
/// plan 0 = [1,1,2,2] establishes districts {1,2}; plan 1 = [1,1,1,1] drops district 2. The run
/// must fail fast rather than emit a null/zero score for the vanished district.
#[test]
fn polsby_popper_fails_when_later_frame_drops_a_first_assignment_district() {
    let plans = vec![vec![1u16, 1, 2, 2], vec![1u16, 1, 1, 1]];
    let f = polsby_fixture(&plans);
    let stderr = run_failure(&[
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

    assert!(
        stderr.contains("districts [2] from the first assignment are missing from a later plan")
            && stderr.contains("same district labels"),
        "stderr should explain the dropped-district schema failure, got: {stderr}"
    );
}

/// `polsby-popper` has an explicit "no rows seen" branch that builds an empty schema-less DataFrame
/// and finishes the parquet writer without ever calling `sorted_district_ids`. Drive that branch
/// with a BEN file containing zero frames and verify the binary exits cleanly and produces a
/// readable parquet.
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

/// MkvChain BEN: `step` advances by `n_reps`, `accepted_count` by 1, and one score row is emitted
/// per frame. Reuses the known scores from `polsby_popper_with_explicit_perimeter_key`.
///
/// Frames after coalescing:
///   frame 1: [1,1,2,2], count=2 → d1=d2=2*PI/9
///   frame 2: [1,2,2,2], count=1 → d1=PI/4, d2=3*PI/16
#[test]
fn polsby_popper_mkvchain_step_advances_by_n_reps() {
    let f = polsby_fixture_mkv(&[
        vec![1, 1, 2, 2],
        vec![1, 1, 2, 2], // coalesces with previous → count=2
        vec![1, 2, 2, 2],
    ]);
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
    assert_eq!(u64_col(&df, "step"), vec![1, 3]);
    assert_eq!(u32_col(&df, "n_reps"), vec![2, 1]);
    assert_eq!(u64_col(&df, "accepted_count"), vec![1, 2]);
}
