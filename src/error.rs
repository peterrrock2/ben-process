//! Crate-wide error type.
//!
//! A single structured enum used throughout the crate. Genuine I/O and malformed-input errors flow
//! in as `io::Error` via `#[from]`; polars/parquet errors as `PolarsError`; and the tool's own
//! contract violations get dedicated, matchable variants.
//!
//! The `Io` and `Parquet` variants are `#[error(transparent)]`, so they display as the underlying
//! error's own message.

use polars::prelude::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenError {
    /// I/O failures and malformed-input errors (file access, graph JSON parsing, CLI validation),
    /// constructed as `io::Error` (usually `InvalidData`/`InvalidInput`).
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Parquet / dataframe errors from polars.
    #[error(transparent)]
    Parquet(#[from] PolarsError),

    /// A decoded BEN assignment's length disagrees with the graph's node count.
    #[error("BEN assignment has {actual} entries but graph has {expected} nodes")]
    AssignmentLength { actual: usize, expected: usize },

    /// A district id is at or beyond the dense `u128`-bitmask limit.
    #[error("district id {id} exceeds current {limit}-district limit; widen the observed bitmask")]
    DistrictLimitExceeded { id: u16, limit: u16 },

    /// A plan's district label set differs from the first plan's (a district was added or
    /// dropped). The composed explanation is carried as the payload.
    #[error("{0}")]
    DistrictSetChanged(String),

    /// A per-accepted-frame mode (changed-assignments) found no frames to process.
    #[error("No data found")]
    NoData,
}

/// Convenience result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, BenError>;

pub(crate) fn invalid_data(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}
