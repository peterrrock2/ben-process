use crate::error::{invalid_data, Error, Result};
use arrow_array::types::ByteArrayType;
use arrow_array::{Array, BinaryArray, GenericByteArray, LargeBinaryArray, RecordBatch};
use geo::{Area, ConvexHull, Coord, MapCoords};
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoparquet::metadata::{GeoParquetColumnMetadata, GeoParquetMetadata};
use geoparquet::reader::GeoParquetReaderBuilder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use std::fs::File;
use wkb::reader::read_wkb;

fn geoparquet_error(message: impl Into<String>) -> Error {
    Error::GeoParquet(message.into())
}

fn geometry_error(message: impl Into<String>) -> Error {
    Error::Geometry(message.into())
}

fn crs_error(message: impl Into<String>) -> Error {
    Error::Crs(message.into())
}

#[derive(Debug)]
pub struct ReockUnit {
    pub area: f64,
    pub convex_hull_points: Vec<Coord<f64>>,
}

#[derive(Debug)]
pub struct ReockGeometries {
    pub units: Vec<ReockUnit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrsStatus {
    Projected,
    Geographic,
    Unknown,
}

#[derive(Debug, Clone, Copy)]
pub struct Circle {
    pub center: Coord<f64>,
    pub radius: f64,
}

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

/// There are numerous CRS classifications, and the GeoParquetColumnMetadata::crs field is
/// documented as adhering to the PROJ JSON specification.
/// [see proj documentation](https://proj.org/en/stable/specifications/projjson.html#high-level-objects)
/// The JSON object will come with "type" which carries information about what kind of CRS is
/// contained in the geoparquet.
fn classify_projjson_crs(crs: &Value) -> CrsStatus {
    match crs.get("type").and_then(|value| value.as_str()) {
        Some("ProjectedCRS") | Some("DerivedProjectedCRS") => CrsStatus::Projected,
        Some("GeographicCRS")
        | Some("GeodeticCRS")
        | Some("DerivedGeographicCRS")
        | Some("DerivedGeodeticCRS") => CrsStatus::Geographic,
        _ => CrsStatus::Unknown,
    }
}

fn classify_geoparquet_crs(crs: Option<&Value>) -> CrsStatus {
    match crs {
        None => CrsStatus::Geographic,
        Some(Value::Null) => CrsStatus::Unknown,
        Some(value) => classify_projjson_crs(value),
    }
}

fn resolve_source_crs(
    source_crs_metadata: Option<&Value>,
    source_crs_override: Option<&str>,
) -> Result<String> {
    if let Some(override_crs) = source_crs_override {
        return Ok(override_crs.to_string());
    };

    match source_crs_metadata {
        None => Ok("OGC:CRS84".to_string()),
        Some(Value::Null) => Err(crs_error(
            "GeoParquet geometry CRS is unknown; pass --source-crs to reproject",
        )),
        Some(metadata_crs) => match classify_projjson_crs(metadata_crs) {
            CrsStatus::Projected | CrsStatus::Geographic => Ok(metadata_crs.to_string()),
            CrsStatus::Unknown => Err(crs_error(
                "GeoParquet geometry CRS is unknown; pass --source-crs to reproject",
            )),
        },
    }
}

fn classify_user_crs(crs: &str) -> Result<CrsStatus> {
    let crs_obj =
        proj::Proj::new(crs).map_err(|e| crs_error(format!("failed to parse CRS {crs:?}: {e}")))?;

    let projjson = crs_obj
        .to_projjson(None, None, None)
        .map_err(|e| crs_error(format!("failed to convert CRS {crs:?} to PROJJSON: {e}")))?;

    let value = serde_json::from_str(&projjson).map_err(|e| {
        crs_error(format!(
            "PROJ returned invalid PROJJSON for CRS {crs:?}: {e}"
        ))
    })?;

    Ok(classify_projjson_crs(&value))
}

pub struct Reprojector {
    transformer: proj::Proj,
    transform_status: CrsStatus,
}

fn build_reprojector(
    source_crs_metadata: Option<&Value>,
    source_crs_override: Option<&str>,
    target_crs: Option<&str>,
) -> Result<Option<Reprojector>> {
    let Some(target_crs) = target_crs else {
        return Ok(None);
    };

    let transform_status = classify_user_crs(target_crs)?;
    if transform_status != CrsStatus::Projected {
        return Err(crs_error(format!(
            "Target CRS '{}' is not a projected CRS",
            target_crs
        )));
    }

    let source_crs = resolve_source_crs(source_crs_metadata, source_crs_override)?;

    let transformer = proj::Proj::new_known_crs(&source_crs, target_crs, None).map_err(|e| {
        crs_error(format!(
            "Failed to create Proj transformer from source CRS '{}' to target CRS '{}': {}",
            source_crs, target_crs, e
        ))
    })?;

    Ok(Some(Reprojector {
        transformer,
        transform_status,
    }))
}

fn classify_effective_source_crs(
    source_crs_metadata: Option<&Value>,
    source_crs_override: Option<&str>,
) -> Result<CrsStatus> {
    match source_crs_override {
        Some(source_crs) => classify_user_crs(source_crs),
        None => Ok(classify_geoparquet_crs(source_crs_metadata)),
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

fn decode_wkb_geometry(wkb: &[u8]) -> Result<Geometry<f64>> {
    let wkb = read_wkb(wkb)
        .map_err(|e| geometry_error(format!("Failed to decode WKB geometry from bytes: {}", e)))?;

    Ok(wkb.to_geometry())
}

fn validate_area(area: f64) -> Result<()> {
    if !area.is_finite() {
        return Err(geometry_error(format!(
            "computed non-finite geometry area {area}"
        )));
    }

    if area <= 0.0 {
        return Err(geometry_error(format!(
            "computed non-positive geometry area {area}"
        )));
    }

    Ok(())
}

fn parse_reock_unit_from_wkb(wkb: &[u8], reprojector: Option<&Reprojector>) -> Result<ReockUnit> {
    let mut geometry = decode_wkb_geometry(wkb)?;

    match &geometry {
        Geometry::Polygon(_) | Geometry::MultiPolygon(_) => {}
        other => {
            return Err(geometry_error(format!(
                "Geometry from WKB is not a Polygon or MultiPolygon: {:?}",
                other
            )));
        }
    }

    if let Some(reprojector) = reprojector {
        geometry = geometry
            .try_map_coords(|Coord { x, y }| {
                reprojector
                    .transformer
                    .convert((x, y))
                    .map(|(x, y)| Coord { x, y })
            })
            .map_err(|e| crs_error(format!("Failed to reproject geometry from WKB: {e}")))?;
    };

    let area = geometry.unsigned_area();

    validate_area(area)?;

    let convex_hull = geometry.convex_hull();

    let mut convex_hull_points = convex_hull
        .exterior()
        .points()
        .map(|p| Coord { x: p.x(), y: p.y() })
        .collect::<Vec<_>>();

    if convex_hull_points.first() == convex_hull_points.last() {
        convex_hull_points.pop();
    }

    if convex_hull_points.len() < 3 {
        return Err(geometry_error(
            "computed convex hull has fewer than 3 distinct points",
        ));
    }

    Ok(ReockUnit {
        area,
        convex_hull_points,
    })
}

fn update_units_with_wkb_binary_array<T>(
    array: &GenericByteArray<T>,
    reprojector: Option<&Reprojector>,
    units: &mut Vec<ReockUnit>,
) -> Result<()>
where
    T: ByteArrayType<Native = [u8]>,
{
    for i in 0..array.len() {
        if array.is_null(i) {
            return Err(geometry_error(format!(
                "Geometry column contains null value at index {}",
                i
            )));
        }

        let wkb = array.value(i);
        let unit = parse_reock_unit_from_wkb(wkb, reprojector)?;
        units.push(unit);
    }

    Ok(())
}

fn update_reock_units_with_batch(
    batch: &RecordBatch,
    geometry_column: &str,
    reprojector: Option<&Reprojector>,
    units: &mut Vec<ReockUnit>,
) -> Result<()> {
    let array = batch.column_by_name(geometry_column).ok_or_else(|| {
        geoparquet_error(format!(
            "Geometry column '{geometry_column}' not found in RecordBatch"
        ))
    })?;

    if let Some(binary) = array.as_any().downcast_ref::<BinaryArray>() {
        update_units_with_wkb_binary_array(binary, reprojector, units)
    } else if let Some(binary) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        update_units_with_wkb_binary_array(binary, reprojector, units)
    } else {
        Err(geometry_error(format!(
            "Geometry column '{geometry_column}' has Arrow type {:?}; expected WKB binary",
            array.data_type()
        )))
    }
}

fn build_reock_geometry(
    record_batch_reader_builder: ParquetRecordBatchReaderBuilder<File>,
    geometry_column: &str,
    reprojector: Option<Reprojector>,
) -> Result<ReockGeometries> {
    let reader = record_batch_reader_builder
        .build()
        .map_err(Error::Parquet)?;

    let mut units = Vec::new();

    for batch in reader {
        let batch = batch.map_err(|e| Error::Arrow(e.to_string()))?;

        update_reock_units_with_batch(&batch, geometry_column, reprojector.as_ref(), &mut units)?;
    }

    Ok(ReockGeometries { units })
}

pub struct ReockLoadOptions<'a> {
    pub geometry_column: Option<&'a str>,
    pub source_crs: Option<&'a str>,
    pub target_crs: Option<&'a str>,
    pub allow_geographic_crs: bool,
    pub allow_unknown_crs: bool,
}

pub fn load_reock_units_from_geoparquet(
    file_path: &str,
    options: ReockLoadOptions<'_>,
) -> Result<ReockGeometries> {
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

    validate_geometry_encoding(&geo_col_meta, &resolved_geometry_column)?;

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

    build_reock_geometry(
        record_batch_reader_builder,
        &resolved_geometry_column,
        reprojector,
    )
}
