//! Library entry point for `ben-process`.
//!
//! The binary parses CLI args and delegates here; mode setup and dispatch live behind this seam so
//! they can be tested and refactored without growing `main.rs`.

pub mod cli;
mod commands;
mod district;
pub mod error;
pub mod graph;
pub mod input;
pub mod metrics;
mod output;
pub mod pipeline;

use cli::Args;

pub fn run(args: Args) -> crate::error::Result<()> {
    commands::run(args)
}
