#[path = "common/mod.rs"]
mod common;

use common::*;

/// Five input frames with three label-invariant partitions:
///   * P_A appears as itself and again with districts {1,2} swapped (label-perm)
///   * P_B appears as itself and again byte-identical
///   * P_C appears once
///
/// Expected: extract-unique-plans writes exactly the 3 first-occurrences, preserving original
/// labels of the first time each partition was seen.
#[test]
fn extract_unique_plans_dedups_label_permutations() {
    let plans: Vec<Vec<u16>> = vec![
        vec![1, 1, 1, 2, 2, 2], // P_A first
        vec![1, 1, 2, 2, 1, 1], // P_B first
        vec![2, 2, 2, 1, 1, 1], // P_A again, labels swapped — should dedup
        vec![1, 1, 2, 2, 1, 1], // P_B again, identical — should dedup
        vec![1, 2, 1, 2, 1, 2], // P_C first
    ];
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

    assert_eq!(
        extracted,
        vec![
            vec![1u16, 1, 1, 2, 2, 2], // P_A first occurrence (original labels)
            vec![1u16, 1, 2, 2, 1, 1], // P_B first occurrence
            vec![1u16, 2, 1, 2, 1, 2], // P_C first occurrence
        ]
    );
}

/// `extract-unique-plans` opens the input via `BenDecoder::new` after a `count_frames` pass, both
/// of which return errors that propagate via `?`. Feed it a non-BEN file and assert the binary
/// exits non-zero rather than silently producing an empty/corrupt output. This guards the error
/// path in src/metrics/extract_unique_plans.rs and src/pipeline.rs (count_frames).
#[test]
fn extract_unique_plans_fails_on_corrupted_ben_input() {
    let tmp = tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let bogus = dir.join("not_a_real.jsonl.ben");
    // Bytes that are definitely not a valid BEN stream — neither the magic header nor any decodable
    // frames.
    std::fs::write(&bogus, b"this is not a BEN file").unwrap();

    let output = Command::new(bin())
        .args([
            "--mode",
            "extract-unique-plans",
            "--ben-file",
            bogus.to_str().unwrap(),
            "--output-dir",
            dir.to_str().unwrap(),
            "--no-progress",
        ])
        .output()
        .expect("failed to spawn ben-process");

    assert!(
        !output.status.success(),
        "extract-unique-plans should fail on a corrupted BEN input; stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
