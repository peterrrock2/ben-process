use super::{crs_error, geometry_error, CrsStatus};
use crate::error::Result;
use geo::{Coord, MapCoords};
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::{Geometry, MultiPolygon};
use serde_json::Value;

pub(super) struct Reprojector {
    pub(super) transformer: proj::Proj,
    pub(super) transform_status: CrsStatus,
}

pub(super) fn classify_projjson_crs(crs: &Value) -> CrsStatus {
    match crs.get("type").and_then(|value| value.as_str()) {
        Some("ProjectedCRS") | Some("DerivedProjectedCRS") => CrsStatus::Projected,
        Some("GeographicCRS")
        | Some("GeodeticCRS")
        | Some("DerivedGeographicCRS")
        | Some("DerivedGeodeticCRS") => CrsStatus::Geographic,
        _ => CrsStatus::Unknown,
    }
}

pub(super) fn classify_geoparquet_crs(crs: Option<&Value>) -> CrsStatus {
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

pub(super) fn classify_user_crs(crs: &str) -> Result<CrsStatus> {
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

pub(super) fn build_reprojector(
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

pub(super) fn classify_effective_source_crs(
    source_crs_metadata: Option<&Value>,
    source_crs_override: Option<&str>,
) -> Result<CrsStatus> {
    match source_crs_override {
        Some(source_crs) => classify_user_crs(source_crs),
        None => Ok(classify_geoparquet_crs(source_crs_metadata)),
    }
}

pub(super) fn decode_wkb_geometry(wkb: &[u8]) -> Result<Geometry<f64>> {
    let wkb = ::wkb::reader::read_wkb(wkb)
        .map_err(|e| geometry_error(format!("Failed to decode WKB geometry from bytes: {}", e)))?;

    Ok(wkb.to_geometry())
}

pub(super) fn geometry_to_multipolygon(geometry: Geometry<f64>) -> Result<MultiPolygon<f64>> {
    match geometry {
        Geometry::Polygon(poly) => Ok(MultiPolygon(vec![poly])),
        Geometry::MultiPolygon(mp) => Ok(mp),
        other => Err(geometry_error(format!(
            "Geometry from WKB is not a Polygon or MultiPolygon: {:?}",
            other
        ))),
    }
}

pub(super) fn reproject_geometry(
    geometry: Geometry<f64>,
    reprojector: Option<&Reprojector>,
) -> Result<Geometry<f64>> {
    let Some(reprojector) = reprojector else {
        return Ok(geometry);
    };

    geometry
        .try_map_coords(|Coord { x, y }| {
            reprojector
                .transformer
                .convert((x, y))
                .map(|(x, y)| Coord { x, y })
        })
        .map_err(|e| crs_error(format!("Failed to reproject geometry from WKB: {e}")))
}
