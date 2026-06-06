use ben_process::cli::Args;
use clap::Parser;

fn main() {
    if let Err(err) = ben_process::run(Args::parse()) {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}
