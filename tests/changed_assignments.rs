#[path = "common/mod.rs"]
mod common;

use common::*;

/// Read the per-node `changed_assignments` column from the mode's Parquet output for
/// `_accept_<n>_`.
fn read_changed_assignments(dir: &std::path::Path, accept_n: u32) -> Vec<f64> {
    let path = dir.join(format!(
        "plans_accept_{accept_n}_changed_assignments.parquet"
    ));
    f64_col(&read_parquet(&path), "changed_assignments")
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
    let df = read_parquet(&f.dir.join("plans_accept_1_changed_assignments.parquet"));
    assert_eq!(u32_col(&df, "node"), vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(f64_col(&df, "changed_assignments"), vec![0.0; 6]);
}

/// Multi-plan changed-assignments: with `--randomize-reassignments` default `false`, output is
/// deterministic.
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
    assert_eq!(
        read_changed_assignments(&f.dir, 3),
        vec![0.0, 2.0, 1.0, 0.0, 1.0, 1.0]
    );
}

/// MkvChain BEN with a repeated assignment collapses into a frame with `count > 1`.
/// `changed-assignments` semantics are per-accepted-record (per-frame), so a 3-sample / 2-frame
/// ensemble should report 2 accepted.
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
    let path = dir.join("plans_accept_2_changed_assignments.parquet");
    assert!(
        path.exists(),
        "output file should use frame count (2), not sample count (3)"
    );
    assert_eq!(
        f64_col(&read_parquet(&path), "changed_assignments"),
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
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
    assert_eq!(
        read_changed_assignments(&f.dir, 3),
        vec![0.0, 1.0, 0.5, 0.0, 0.5, 0.5]
    );
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
    assert_eq!(
        read_changed_assignments(&f.dir, 2),
        vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    );
}

/// A `--max-accepted` larger than the file's frame count means "everything": the normalization
/// divisor and the `_accept_N_` output filename must both use the 3 frames actually present, not
/// the requested 10. (Previously the divisor was `max_accepted - 1` = 9, silently deflating every
/// normalized value; expected values here match `changed_assignments_tri_plans_normalized`.)
#[test]
fn changed_assignments_max_accepted_beyond_frame_count_normalizes_by_actual_frames() {
    let f = fixture(&tri_plans());
    run(&[
        "--mode",
        "changed-assignments",
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--max-accepted",
        "10",
        "--normalize",
        "--no-progress",
    ]);
    assert_eq!(
        read_changed_assignments(&f.dir, 3),
        vec![0.0, 1.0, 0.5, 0.0, 0.5, 0.5]
    );
}

/// `--randomize-reassignments` uses a thread-local OS-seeded RNG, so we can't pin exact dif counts.
/// This is a smoke test: confirm the binary completes successfully, the output file uses the
/// expected frame count, and the per-unit values are within the valid range. With two
/// label-permuted frames, each per-unit count is either 0 (RNG fired and labels were swapped before
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

    let counts = read_changed_assignments(&f.dir, 2);
    assert_eq!(counts.len(), 4);
    for v in counts {
        assert!(
            v == 0.0 || v == 1.0,
            "per-unit count under --randomize-reassignments must be 0 or 1, got {v}"
        );
    }
}

/// `--seed` makes `--randomize-reassignments` reproducible: two runs with the same seed must
/// produce identical per-node counts. Without a seed the RNG is OS-seeded and runs can differ.
#[test]
fn changed_assignments_seed_makes_randomization_reproducible() {
    let plans = vec![vec![1u16, 1, 2, 2], vec![2u16, 2, 1, 1]];
    let f = fixture(&plans);
    let dir_a = f.dir.join("a");
    let dir_b = f.dir.join("b");
    std::fs::create_dir_all(&dir_a).unwrap();
    std::fs::create_dir_all(&dir_b).unwrap();

    for out in [&dir_a, &dir_b] {
        run(&[
            "--mode",
            "changed-assignments",
            "--ben-file",
            f.ben.to_str().unwrap(),
            "--output-dir",
            out.to_str().unwrap(),
            "--randomize-reassignments",
            "--seed",
            "42",
            "--no-progress",
        ]);
    }

    let a = read_changed_assignments(&dir_a, 2);
    let b = read_changed_assignments(&dir_b, 2);
    assert_eq!(a, b, "same seed must produce identical randomized output");
}

/// changed-assignments enforces the same fixed-district-set invariant as the pipeline modes: a
/// later plan that introduces a new district id must fail fast. plan 0 = [1,1,2,2] establishes
/// {1,2}; plan 1 = [1,3,2,2] introduces district 3.
#[test]
fn changed_assignments_fails_when_later_frame_adds_a_district() {
    let plans = vec![vec![1u16, 1, 2, 2], vec![1u16, 3, 2, 2]];
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
        stderr.contains("encountered districts [3] not present in first assignment")
            && stderr.contains("same district labels"),
        "stderr should explain the changed-district-set failure, got: {stderr}"
    );
}

/// The dropped-district direction too: plan 0 = [1,1,2,2] establishes {1,2}; plan 1 = [1,1,1,1]
/// drops district 2. Its labels are within the first plan's range, so only the fixed-set check
/// (not the permutation range check) catches it.
#[test]
fn changed_assignments_fails_when_later_frame_drops_a_district() {
    let plans = vec![vec![1u16, 1, 2, 2], vec![1u16, 1, 1, 1]];
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
        stderr.contains("districts [2] from the first assignment are missing from a later plan")
            && stderr.contains("same district labels"),
        "stderr should explain the dropped-district failure, got: {stderr}"
    );
}

/// Mixed assignment lengths within one BEN file are a corrupt ensemble: the per-node diff would
/// silently zip-truncate to the shorter frame. The driver's uniform-length contract must reject
/// the file instead.
#[test]
fn changed_assignments_fails_on_mixed_assignment_lengths() {
    let plans = vec![vec![1u16, 1, 2, 2, 1, 1], vec![1u16, 1, 2, 2]];
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
        stderr.contains("assignment length changed from 6 to 4 within the BEN file"),
        "stderr should explain the mixed-length failure, got: {stderr}"
    );
}
