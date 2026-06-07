#[path = "common/mod.rs"]
mod common;

use common::*;

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

/// Five frames, every one a distinct partition (not just a label permutation of any other) → unique
/// count equals frame count.
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
    // Same fixture as extract_unique_plans: 3 distinct partitions among 5 frames (P_A, P_B,
    // P_A-relabeled, P_B-duplicate, P_C).
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
