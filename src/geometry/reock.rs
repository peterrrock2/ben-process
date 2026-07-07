use super::wkb::{decode_wkb_geometry, geometry_to_multipolygon, reproject_geometry, Reprojector};
use super::{geometry_error, validate_area};
use crate::error::{Error, Result};
use arrow_array::types::ByteArrayType;
use arrow_array::{Array, BinaryArray, GenericByteArray, LargeBinaryArray, RecordBatch};
use geo::{Area, ConvexHull, Coord};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

#[derive(Debug)]
pub struct ReockUnit {
    pub area: f64,
    pub convex_hull_points: Vec<Coord<f64>>,
}

#[derive(Debug)]
pub struct ReockGeometries {
    pub units: Vec<ReockUnit>,
}

fn parse_reock_unit_from_wkb(wkb: &[u8], reprojector: Option<&Reprojector>) -> Result<ReockUnit> {
    let geometry = reproject_geometry(decode_wkb_geometry(wkb)?, reprojector)?;
    let geometry = geometry_to_multipolygon(geometry)?;

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
        super::geoparquet_error(format!(
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

pub(super) fn build_reock_geometry(
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
