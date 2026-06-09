use ben_process::cli::Args;
use clap::Parser;
use std::io::Write;

fn main() {
    let args = Args::parse();
    init_logging(args.quiet);

    if let Err(err) = ben_process::run(args) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

/// Initialize status logging. By default, `info`-level status lines print to stderr as plain
/// messages (matching the tool's historical output); `--quiet` turns them off. `RUST_LOG` overrides
/// the level in either case. The final `Error:` line above is a plain `eprintln!` so it always
/// shows, even under `--quiet`.
fn init_logging(quiet: bool) {
    let default_level = if quiet { "off" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(default_level))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();
}
