#[path = "common/mod.rs"]
mod common;

use common::*;

/// Verbosity model (matching binary-ensemble 1.0): status logging is off by default and is enabled
/// by `-v/--verbose`; `-q/--quiet` only suppresses the progress bar. So a default run prints no
/// "Done!" status line, and a `-v` run does. `RUST_LOG` is cleared so the test doesn't depend on
/// the caller's environment. `-q` keeps the progress bar out of captured stderr in both runs.
#[test]
fn verbose_controls_status_output() {
    let f = fixture(&tri_plans());
    let base_args = [
        "cut-edges",
        "--graph-file",
        f.graph.to_str().unwrap(),
        "--ben-file",
        f.ben.to_str().unwrap(),
        "--output-dir",
        f.dir.to_str().unwrap(),
        "-q",
    ];

    let quiet = Command::new(bin())
        .args(base_args)
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to spawn ben-process");
    assert!(quiet.status.success(), "default run should succeed");
    let quiet_stderr = String::from_utf8_lossy(&quiet.stderr);
    assert!(
        quiet_stderr.is_empty(),
        "without -v there should be no status output, got: {quiet_stderr}"
    );

    let verbose = Command::new(bin())
        .args(base_args)
        .arg("-v")
        .env_remove("RUST_LOG")
        .output()
        .expect("failed to spawn ben-process");
    assert!(verbose.status.success(), "verbose run should still succeed");
    let verbose_stderr = String::from_utf8_lossy(&verbose.stderr);
    assert!(
        verbose_stderr.contains("Done!"),
        "-v should print status lines to stderr, got: {verbose_stderr}"
    );
}
