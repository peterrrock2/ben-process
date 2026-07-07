use super::polsby_popper::{build_polsby_popper_geometry, PolsbyPopperGeometries};
use super::reock::{build_reock_geometry, ReockGeometries};
use super::wkb::{build_reprojector, classify_effective_source_crs, Reprojector};
use super::{crs_error, geoparquet_error, CrsStatus};
use crate::error::{invalid_data, Error, Result};
use geoparquet::metadata::{GeoParquetColumnMetadata, GeoParquetMetadata};
use geoparquet::reader::GeoParquetReaderBuilder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

fn open_parquet_file(file_path: &str) -> Result<ParquetRecordBatchReaderBuilder<File>> {
    let file = File::open(file_path)
        .map_err(|e| invalid_data(format!("Failed to open Parquet file {}: {}", file_path, e)))?;

    ParquetRecordBatchReaderBuilder::try_new(file).map_err(Error::Parquet)
}

fn resolve_geometry_column(
    geo_meta: &GeoParquetMetadata,
    geometry_column: Option<&str>,
) -> Result<String> {
    match geometry_column {
        Some(col) => {
            if geo_meta.columns.contains_key(col) {
                Ok(col.to_string())
            } else {
                Err(geoparquet_error(format!(
                    "Geometry column '{}' not found in GeoParquet metadata",
                    col
                )))
            }
        }
        None => {
            let primary_geom_col = geo_meta.primary_column.as_str();
            if geo_meta.columns.contains_key(primary_geom_col) {
                Ok(primary_geom_col.to_string())
            } else {
                Err(geoparquet_error(format!(
                    "Primary geometry column '{}' not found in GeoParquet metadata",
                    primary_geom_col
                )))
            }
        }
    }
}

fn require_geo_column_metadata<'a>(
    geo_meta: &'a GeoParquetMetadata,
    geometry_column: &str,
) -> Result<&'a GeoParquetColumnMetadata> {
    geo_meta.columns.get(geometry_column).ok_or_else(|| {
        geoparquet_error(format!(
            "Geometry column '{}' not found in GeoParquet metadata",
            geometry_column
        ))
    })
}

fn validate_geometry_encoding(
    geo_col_meta: &GeoParquetColumnMetadata,
    geometry_column: &str,
) -> Result<()> {
    match geo_col_meta.encoding {
        geoparquet::metadata::GeoParquetColumnEncoding::WKB => Ok(()),
        _ => Err(geoparquet_error(format!(
            "Geometry column '{}' has unsupported encoding: {:?}",
            geometry_column, geo_col_meta.encoding
        ))),
    }
}

fn validate_effective_crs(
    effective_target_crs: CrsStatus,
    allow_geographic_crs: bool,
    allow_unknown_crs: bool,
) -> Result<()> {
    match effective_target_crs {
        CrsStatus::Projected => Ok(()),
        CrsStatus::Geographic => {
            if allow_geographic_crs {
                Ok(())
            } else {
                Err(crs_error(
                    "effective geometry CRS is geographic; pass --allow-geographic-crs to use it",
                ))
            }
        }
        CrsStatus::Unknown => {
            if allow_unknown_crs {
                Ok(())
            } else {
                Err(crs_error(format!(
                    "effective geometry CRS is {:?}; pass --allow-unknown-crs to use it",
                    effective_target_crs
                )))
            }
        }
    }
}

pub struct ReockLoadOptions<'a> {
    pub geometry_column: Option<&'a str>,
    pub source_crs: Option<&'a str>,
    pub target_crs: Option<&'a str>,
    pub allow_geographic_crs: bool,
    pub allow_unknown_crs: bool,
}

pub type PolsbyPopperGeometryLoadOptions<'a> = ReockLoadOptions<'a>;

fn prepare_geoparquet_geometry_reader(
    file_path: &str,
    options: ReockLoadOptions<'_>,
) -> Result<(
    ParquetRecordBatchReaderBuilder<File>,
    String,
    Option<Reprojector>,
)> {
    let ReockLoadOptions {
        geometry_column,
        source_crs,
        target_crs,
        allow_geographic_crs,
        allow_unknown_crs,
    } = options;

    let record_batch_reader_builder = open_parquet_file(file_path)?;

    let geo_meta = record_batch_reader_builder
        .geoparquet_metadata()
        .ok_or_else(|| geoparquet_error("No GeoParquet metadata found in the Parquet file"))?
        .map_err(|e| geoparquet_error(format!("Failed to read GeoParquet metadata: {}", e)))?;

    let resolved_geometry_column = resolve_geometry_column(&geo_meta, geometry_column)?;
    let geo_col_meta = require_geo_column_metadata(&geo_meta, &resolved_geometry_column)?;

    validate_geometry_encoding(geo_col_meta, &resolved_geometry_column)?;

    let reprojector = build_reprojector(geo_col_meta.crs.as_ref(), source_crs, target_crs)?;

    let effective_target_crs = match &reprojector {
        Some(r) => r.transform_status,
        None => classify_effective_source_crs(geo_col_meta.crs.as_ref(), source_crs)?,
    };

    validate_effective_crs(
        effective_target_crs,
        allow_geographic_crs,
        allow_unknown_crs,
    )?;

    Ok((
        record_batch_reader_builder,
        resolved_geometry_column,
        reprojector,
    ))
}

pub fn load_reock_units_from_geoparquet(
    file_path: &str,
    options: ReockLoadOptions<'_>,
) -> Result<ReockGeometries> {
    let (record_batch_reader_builder, resolved_geometry_column, reprojector) =
        prepare_geoparquet_geometry_reader(file_path, options)?;

    build_reock_geometry(
        record_batch_reader_builder,
        &resolved_geometry_column,
        reprojector,
    )
}

pub fn load_polsby_popper_geometry_from_geoparquet(
    file_path: &str,
    options: PolsbyPopperGeometryLoadOptions<'_>,
    graph_edges: &[(u32, u32)],
    graph_node_count: usize,
) -> Result<PolsbyPopperGeometries> {
    let (record_batch_reader_builder, resolved_geometry_column, reprojector) =
        prepare_geoparquet_geometry_reader(file_path, options)?;

    build_polsby_popper_geometry(
        record_batch_reader_builder,
        &resolved_geometry_column,
        reprojector,
        graph_edges,
        graph_node_count,
    )
}
