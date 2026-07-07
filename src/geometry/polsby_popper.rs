use super::wkb::{decode_wkb_geometry, geometry_to_multipolygon, reproject_geometry, Reprojector};
use super::{geometry_error, geoparquet_error, validate_area, validate_length};
use crate::error::{Error, Result};
use arrow_array::types::ByteArrayType;
use arrow_array::{Array, BinaryArray, GenericByteArray, LargeBinaryArray, RecordBatch};
use geo::{Area, BooleanOps, Coord};
use geo_types::{MultiPolygon, Polygon};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

#[derive(Debug)]
pub struct PolsbyPopperGeometries {
    pub area_values: Vec<f64>,
    pub total_perimeter_values: Vec<f64>,
    pub shared_perimeters: Vec<f64>,
}

#[derive(Debug)]
struct PolsbyPopperUnit {
    geometry: MultiPolygon<f64>,
    area: f64,
    perimeter: f64,
}

fn parse_polsby_popper_unit_from_wkb(
    wkb: &[u8],
    reprojector: Option<&Reprojector>,
) -> Result<PolsbyPopperUnit> {
    let geometry = reproject_geometry(decode_wkb_geometry(wkb)?, reprojector)?;
    let geometry = geometry_to_multipolygon(geometry)?;
    let area = geometry.unsigned_area();
    let perimeter = multipolygon_perimeter(&geometry);

    validate_area(area)?;
    validate_length("perimeter", perimeter)?;
    if perimeter <= 0.0 {
        return Err(geometry_error("computed non-positive geometry perimeter"));
    }

    Ok(PolsbyPopperUnit {
        geometry,
        area,
        perimeter,
    })
}

fn coord_distance(a: Coord<f64>, b: Coord<f64>) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn ring_perimeter(coords: &[Coord<f64>]) -> f64 {
    coords
        .windows(2)
        .map(|window| coord_distance(window[0], window[1]))
        .sum()
}

fn polygon_perimeter(poly: &Polygon<f64>) -> f64 {
    ring_perimeter(&poly.exterior().0)
        + poly
            .interiors()
            .iter()
            .map(|ring| ring_perimeter(&ring.0))
            .sum::<f64>()
}

pub(crate) fn multipolygon_perimeter(mp: &MultiPolygon<f64>) -> f64 {
    mp.0.iter().map(polygon_perimeter).sum()
}

fn ring_segments(coords: &[Coord<f64>], segments: &mut Vec<(Coord<f64>, Coord<f64>)>) {
    for window in coords.windows(2) {
        if coord_distance(window[0], window[1]) > 0.0 {
            segments.push((window[0], window[1]));
        }
    }
}

fn multipolygon_boundary_segments(mp: &MultiPolygon<f64>) -> Vec<(Coord<f64>, Coord<f64>)> {
    let mut segments = Vec::new();
    for poly in &mp.0 {
        ring_segments(&poly.exterior().0, &mut segments);
        for interior in poly.interiors() {
            ring_segments(&interior.0, &mut segments);
        }
    }
    segments
}

fn signed_segment_position(point: Coord<f64>, start: Coord<f64>, unit: Coord<f64>) -> f64 {
    (point.x - start.x) * unit.x + (point.y - start.y) * unit.y
}

fn shared_segment_length(a0: Coord<f64>, a1: Coord<f64>, b0: Coord<f64>, b1: Coord<f64>) -> f64 {
    let ax = a1.x - a0.x;
    let ay = a1.y - a0.y;
    let a_len = (ax * ax + ay * ay).sqrt();
    if a_len == 0.0 {
        return 0.0;
    }

    let cross0 = ax * (b0.y - a0.y) - ay * (b0.x - a0.x);
    let cross1 = ax * (b1.y - a0.y) - ay * (b1.x - a0.x);
    if cross0.abs() > 1e-8 || cross1.abs() > 1e-8 {
        return 0.0;
    }

    let unit = Coord {
        x: ax / a_len,
        y: ay / a_len,
    };
    let b0_pos = signed_segment_position(b0, a0, unit);
    let b1_pos = signed_segment_position(b1, a0, unit);
    let overlap_start = 0.0f64.max(b0_pos.min(b1_pos));
    let overlap_end = a_len.min(b0_pos.max(b1_pos));

    (overlap_end - overlap_start).max(0.0)
}

pub(crate) fn shared_boundary_length(left: &MultiPolygon<f64>, right: &MultiPolygon<f64>) -> f64 {
    let left_segments = multipolygon_boundary_segments(left);
    let right_segments = multipolygon_boundary_segments(right);
    let mut length = 0.0;

    for &(a0, a1) in &left_segments {
        for &(b0, b1) in &right_segments {
            length += shared_segment_length(a0, a1, b0, b1);
        }
    }

    length
}

fn update_polsby_popper_units_with_wkb_binary_array<T>(
    array: &GenericByteArray<T>,
    reprojector: Option<&Reprojector>,
    units: &mut Vec<PolsbyPopperUnit>,
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
        let unit = parse_polsby_popper_unit_from_wkb(wkb, reprojector)?;
        units.push(unit);
    }

    Ok(())
}

fn update_polsby_popper_units_with_batch(
    batch: &RecordBatch,
    geometry_column: &str,
    reprojector: Option<&Reprojector>,
    units: &mut Vec<PolsbyPopperUnit>,
) -> Result<()> {
    let array = batch.column_by_name(geometry_column).ok_or_else(|| {
        geoparquet_error(format!(
            "Geometry column '{geometry_column}' not found in RecordBatch"
        ))
    })?;

    if let Some(binary) = array.as_any().downcast_ref::<BinaryArray>() {
        update_polsby_popper_units_with_wkb_binary_array(binary, reprojector, units)
    } else if let Some(binary) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        update_polsby_popper_units_with_wkb_binary_array(binary, reprojector, units)
    } else {
        Err(geometry_error(format!(
            "Geometry column '{geometry_column}' has Arrow type {:?}; expected WKB binary",
            array.data_type()
        )))
    }
}

pub(super) fn build_polsby_popper_geometry(
    record_batch_reader_builder: ParquetRecordBatchReaderBuilder<File>,
    geometry_column: &str,
    reprojector: Option<Reprojector>,
    graph_edges: &[(u32, u32)],
    graph_node_count: usize,
) -> Result<PolsbyPopperGeometries> {
    let reader = record_batch_reader_builder
        .build()
        .map_err(Error::Parquet)?;

    let mut units = Vec::new();

    for batch in reader {
        let batch = batch.map_err(|e| Error::Arrow(e.to_string()))?;

        update_polsby_popper_units_with_batch(
            &batch,
            geometry_column,
            reprojector.as_ref(),
            &mut units,
        )?;
    }

    if units.len() != graph_node_count {
        return Err(Error::AssignmentLength {
            actual: units.len(),
            actual_label: "geometry row count",
            expected: graph_node_count,
            expected_label: "graph node count",
        });
    }

    let mut shared_perimeters = Vec::with_capacity(graph_edges.len());
    let mut shared_by_node = vec![0.0; units.len()];
    for &(node_u, node_v) in graph_edges {
        let u = node_u as usize;
        let v = node_v as usize;
        let overlap_area = units[u]
            .geometry
            .intersection(&units[v].geometry)
            .unsigned_area();
        if overlap_area > 1e-8 {
            return Err(geometry_error(format!(
                "graph edge ({node_u}, {node_v}) has geometries that overlap by area {overlap_area}"
            )));
        }

        let shared = shared_boundary_length(&units[u].geometry, &units[v].geometry);
        if shared <= 1e-8 {
            return Err(geometry_error(format!(
                "graph edge ({node_u}, {node_v}) has no shared geometry boundary"
            )));
        }
        shared_perimeters.push(shared);
        shared_by_node[u] += shared;
        shared_by_node[v] += shared;
    }

    for (node, unit) in units.iter().enumerate() {
        if shared_by_node[node] > unit.perimeter + 1e-8 {
            return Err(geometry_error(format!(
                "shared boundary lengths for node {node} exceed its perimeter"
            )));
        }
    }

    Ok(PolsbyPopperGeometries {
        area_values: units.iter().map(|unit| unit.area).collect(),
        total_perimeter_values: units.iter().map(|unit| unit.perimeter).collect(),
        shared_perimeters,
    })
}
