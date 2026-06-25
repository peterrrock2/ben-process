use ben_process::cli::Cli;
use clap::Parser;
use std::io::Write;

fn main() {
    let cli = Cli::parse();
    init_logging(cli.verbose);

    if let Err(err) = ben_process::run(cli) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

/// Initialize status logging. Quiet by default: only `warn`-level diagnostics (and the fatal
/// `Error:` line) print. `--verbose` raises the default to `info` so status lines show. `RUST_LOG`
/// overrides the level in either case. The `Error:` line above is a plain `eprintln!` so it always
/// shows regardless of level.
fn init_logging(verbose: bool) {
    let default_level = if verbose { "info" } else { "warn" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();
}
