#[path = "common/mod.rs"]
mod common;

use common::*;

/// Without `--quiet` the binary prints status lines (e.g. "Done!") to stderr; with `--quiet` they
/// are suppressed while the run still succeeds. `RUST_LOG` is cleared so the test doesn't depend on
/// the caller's environment.
#[test]
fn quiet_suppresses_status_output() {
    let f = fixture(&tri_plans());
    let base_args = [
        "--mode",
        "cut-edges",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "--no-progress",
    ];

    let loud = Command::new(bin())
        .args(base_args)
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to spawn ben-process");
    assert!(loud.status.success(), "loud run should succeed");
    let loud_stderr = String::from_utf8_lossy(&loud.stderr);
    assert!(
        loud_stderr.contains("Done!"),
        "default run should print status lines to stderr, got: {loud_stderr}"
    );

    let quiet = Command::new(bin())
        .args(base_args)
        .arg("--quiet")
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to spawn ben-process");
    assert!(quiet.status.success(), "quiet run should still succeed");
    let quiet_stderr = String::from_utf8_lossy(&quiet.stderr);
    assert!(
        quiet_stderr.is_empty(),
        "--quiet should suppress all status output, got: {quiet_stderr}"
    );
}
