#[path = "common/mod.rs"]
mod common;

use common::*;

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
    let body =
        std::fs::read_to_string(f.dir.join("plans_accept_3_changed_assignments.txt")).unwrap();
    assert_eq!(body, "[0.0, 2.0, 1.0, 0.0, 1.0, 1.0]\nTotal Accepted: 3");
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
    let body = std::fs::read_to_string(dir.join("plans_accept_2_changed_assignments.txt"))
        .expect("output file should use frame count (2), not sample count (3)");
    assert_eq!(body, "[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]\nTotal Accepted: 2");
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

#[test]
fn changed_assignments_reports_later_unseen_assignment_labels() {
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
        stderr.contains("encountered assignment label 3 at index 1")
            && stderr.contains("outside first assignment label range"),
        "stderr should explain unseen assignment labels, got: {stderr}"
    );
}
