//! Library entry point for `ben-process`.
//!
//! The binary parses CLI args and delegates here; mode setup and dispatch live
//! behind this seam so they can be tested and refactored without growing
//! `main.rs`.

pub mod changed_assignments;
pub mod cli;
mod commands;
pub mod graph;
pub mod metrics;
pub mod pipeline;

use cli::Args;

pub fn run(args: Args) -> std::result::Result<(), Box<dyn std::error::Error>> {
    commands::run(args)
}
