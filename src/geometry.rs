use crate::error::{Error, Result};

mod geoparquet;
mod polsby_popper;
mod reock;
mod wkb;

pub use geoparquet::{
    load_polsby_popper_geometry_from_geoparquet, load_reock_units_from_geoparquet,
    PolsbyPopperGeometryLoadOptions, ReockLoadOptions,
};
pub use polsby_popper::PolsbyPopperGeometries;
pub use reock::{ReockGeometries, ReockUnit};

fn geoparquet_error(message: impl Into<String>) -> Error {
    Error::GeoParquet(message.into())
}

fn geometry_error(message: impl Into<String>) -> Error {
    Error::Geometry(message.into())
}

fn crs_error(message: impl Into<String>) -> Error {
    Error::Crs(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CrsStatus {
    Projected,
    Geographic,
    Unknown,
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

fn validate_length(label: &str, length: f64) -> Result<()> {
    if !length.is_finite() {
        return Err(geometry_error(format!(
            "computed non-finite {label} {length}"
        )));
    }
    if length < 0.0 {
        return Err(geometry_error(format!(
            "computed negative {label} {length}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::polsby_popper::{multipolygon_perimeter, shared_boundary_length};
    use super::*;
    use arrow_array::{ArrayRef, BinaryArray, RecordBatch};
    use geo_types::{polygon, LineString, MultiPolygon, Polygon};
    use parquet::arrow::ArrowWriter;
    use parquet::file::metadata::KeyValue;
    use parquet::file::properties::WriterProperties;
    use std::fs::File;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn square(x: f64, y: f64) -> MultiPolygon<f64> {
        MultiPolygon(vec![polygon![
            (x: x, y: y),
            (x: x + 1.0, y: y),
            (x: x + 1.0, y: y + 1.0),
            (x: x, y: y + 1.0),
            (x: x, y: y),
        ]])
    }

    fn square_wkb(x: f64, y: f64) -> Vec<u8> {
        polygon_wkb(&[
            (x, y),
            (x + 1.0, y),
            (x + 1.0, y + 1.0),
            (x, y + 1.0),
            (x, y),
        ])
    }

    fn polygon_wkb(points: &[(f64, f64)]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(1);
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(points.len() as u32).to_le_bytes());
        for &(x, y) in points {
            bytes.extend_from_slice(&x.to_le_bytes());
            bytes.extend_from_slice(&y.to_le_bytes());
        }
        bytes
    }

    fn write_test_geoparquet(rows: Vec<Vec<u8>>) -> NamedTempFile {
        let file = NamedTempFile::new().expect("create temp GeoParquet file");
        let row_refs = rows.iter().map(Vec::as_slice).collect();
        let geometry = Arc::new(BinaryArray::from_vec(row_refs)) as ArrayRef;
        let batch =
            RecordBatch::try_from_iter(vec![("geometry", geometry)]).expect("build record batch");
        let geo_metadata = r#"{
            "version": "1.0.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["Polygon"],
                    "crs": null
                }
            }
        }"#;
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(vec![KeyValue::new(
                "geo".to_string(),
                geo_metadata.to_string(),
            )]))
            .build();
        let output = File::create(file.path()).expect("open temp GeoParquet file");
        let mut writer =
            ArrowWriter::try_new(output, batch.schema(), Some(props)).expect("create writer");

        writer.write(&batch).expect("write record batch");
        writer.close().expect("close writer");
        file
    }

    fn test_load_options() -> ReockLoadOptions<'static> {
        ReockLoadOptions {
            geometry_column: None,
            source_crs: None,
            target_crs: None,
            allow_geographic_crs: true,
            allow_unknown_crs: true,
        }
    }

    #[test]
    fn geoparquet_loaders_read_wkb_polygons_from_public_path() {
        let file = write_test_geoparquet(vec![square_wkb(0.0, 0.0), square_wkb(1.0, 0.0)]);
        let path = file.path().to_str().expect("temp path is utf-8");

        let reock = load_reock_units_from_geoparquet(path, test_load_options()).unwrap();
        assert_eq!(reock.units.len(), 2);
        assert!((reock.units[0].area - 1.0).abs() < 1e-12);
        assert!((reock.units[1].area - 1.0).abs() < 1e-12);
        assert_eq!(reock.units[0].convex_hull_points.len(), 4);
        assert_eq!(reock.units[1].convex_hull_points.len(), 4);

        let polsby =
            load_polsby_popper_geometry_from_geoparquet(path, test_load_options(), &[(0, 1)], 2)
                .unwrap();
        assert_eq!(polsby.area_values, vec![1.0, 1.0]);
        assert_eq!(polsby.total_perimeter_values, vec![4.0, 4.0]);
        assert_eq!(polsby.shared_perimeters, vec![1.0]);
    }

    #[test]
    fn shared_boundary_length_counts_split_matching_edges() {
        let left = square(0.0, 0.0);
        let right = MultiPolygon(vec![Polygon::new(
            LineString::from(vec![
                (1.0, 0.0),
                (2.0, 0.0),
                (2.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.5),
                (1.0, 0.0),
            ]),
            vec![],
        )]);

        assert!((shared_boundary_length(&left, &right) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn shared_boundary_length_ignores_point_touches() {
        let left = square(0.0, 0.0);
        let right = square(1.0, 1.0);

        assert_eq!(shared_boundary_length(&left, &right), 0.0);
    }

    #[test]
    fn multipolygon_perimeter_includes_holes() {
        let poly = Polygon::new(
            LineString::from(vec![
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 4.0),
                (0.0, 4.0),
                (0.0, 0.0),
            ]),
            vec![LineString::from(vec![
                (1.0, 1.0),
                (2.0, 1.0),
                (2.0, 2.0),
                (1.0, 2.0),
                (1.0, 1.0),
            ])],
        );

        assert_eq!(multipolygon_perimeter(&MultiPolygon(vec![poly])), 20.0);
    }
}
