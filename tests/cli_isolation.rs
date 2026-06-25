#[path = "common/mod.rs"]
mod common;

use common::*;

// Each subcommand exposes only its own flags, so a flag owned by another mode must be rejected by
// the parser rather than silently ignored. The BEN path need not exist: clap rejects the unknown
// flag before any file is opened.

/// `unique-plans` is graph-free and has no `--keys`.
#[test]
fn unique_plans_rejects_keys_flag() {
    let stderr = run_failure(&[
        "unique-plans",
        "-b",
        "nonexistent.jsonl.ben",
        "--keys",
        "pop",
    ]);
    assert!(
        stderr.contains("--keys"),
        "unique-plans should reject --keys, got: {stderr}"
    );
}

/// `changed-assignments` is frame-based and has no `--max-samples` (it uses `--max-accepted`).
#[test]
fn changed_assignments_rejects_max_samples_flag() {
    let stderr = run_failure(&[
        "changed-assignments",
        "-b",
        "nonexistent.jsonl.ben",
        "--max-samples",
        "10",
    ]);
    assert!(
        stderr.contains("--max-samples"),
        "changed-assignments should reject --max-samples, got: {stderr}"
    );
}
