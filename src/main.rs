use ben_process::cli::Args;
use clap::Parser;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    ben_process::run(Args::parse())
}
