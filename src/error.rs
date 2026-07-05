//! Crate-wide error type.
//!
//! A single structured enum used throughout the crate. Genuine I/O errors flow in as `io::Error`
//! via `#[from]`; Parquet and dataframe errors keep their own buckets; and the tool's own contract
//! violations get dedicated, matchable variants.
//!
//! The transparent variants display as the underlying error's own message.

use parquet::errors::ParquetError;
use polars::prelude::PolarsError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    /// I/O failures and legacy malformed-input errors still represented as `io::Error`.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Low-level Parquet reader/writer errors.
    #[error(transparent)]
    Parquet(#[from] ParquetError),

    /// Arrow record-batch/schema errors surfaced while reading columnar data.
    #[error("Arrow error: {0}")]
    Arrow(String),

    /// Dataframe errors from Polars.
    #[error(transparent)]
    Polars(#[from] PolarsError),

    /// GeoParquet metadata or geometry-column contract violations.
    #[error("GeoParquet error: {0}")]
    GeoParquet(String),

    /// Geometry decoding, validation, or metric-preprocessing failures.
    #[error("geometry error: {0}")]
    Geometry(String),

    /// CRS parsing, validation, or reprojection failures.
    #[error("CRS error: {0}")]
    Crs(String),

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
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid_data(message: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, message.into())
}
