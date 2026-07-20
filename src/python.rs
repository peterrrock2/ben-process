use crate::district::{sorted_district_ids, validate_district_set_unchanged};
use crate::geometry::{PolsbyPopperGeometries, ReockGeometries, WkbGeometryLoadOptions};
use crate::metrics::polsby_popper::PreparedPolsbyPopper;
use crate::metrics::reock::PreparedReock;
use crate::metrics::tally_keys::PreparedTally;
use crate::metrics::PreparedMetricOutput;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

enum RustBackedMetric {
    Tally(PreparedTally),
    Reock(PreparedReock),
    PolsbyPopper(PreparedPolsbyPopper),
}

impl RustBackedMetric {
    fn score(&self, assignment: &[u16]) -> crate::error::Result<PreparedMetricOutput> {
        match self {
            Self::Tally(metric) => metric.score_assignment(assignment),
            Self::Reock(metric) => metric.score_assignment(assignment),
            Self::PolsbyPopper(metric) => metric.score_assignment(assignment),
        }
    }
}

#[pyclass(name = "RustBackendScorer", module = "ben_process._rust_backend")]
struct RustBackendScorer {
    metrics: Vec<RustBackedMetric>,
}

#[pymethods]
impl RustBackendScorer {
    #[new]
    fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    fn add_tally(&mut self, columns: Vec<Vec<f64>>) -> PyResult<()> {
        let metric = PreparedTally::new(columns).map_err(value_error)?;
        self.metrics.push(RustBackedMetric::Tally(metric));
        Ok(())
    }

    #[pyo3(signature = (
        rows,
        source_crs=None,
        target_crs=None,
        allow_geographic_crs=false,
        allow_unknown_crs=false,
    ))]
    fn add_reock(
        &mut self,
        py: Python<'_>,
        rows: Vec<Vec<u8>>,
        source_crs: Option<String>,
        target_crs: Option<String>,
        allow_geographic_crs: bool,
        allow_unknown_crs: bool,
    ) -> PyResult<()> {
        let result = py.detach(|| {
            ReockGeometries::from_wkb(
                &rows,
                WkbGeometryLoadOptions {
                    source_crs: source_crs.as_deref(),
                    target_crs: target_crs.as_deref(),
                    allow_geographic_crs,
                    allow_unknown_crs,
                },
            )
            .map(PreparedReock::new)
            .map_err(|error| error.to_string())
        });
        self.metrics.push(RustBackedMetric::Reock(
            result.map_err(PyValueError::new_err)?,
        ));
        Ok(())
    }

    fn add_polsby_popper_graph(
        &mut self,
        area_values: Vec<f64>,
        total_perimeter_values: Option<Vec<f64>>,
        boundary_perimeter_values: Option<Vec<f64>>,
        edges: Vec<(u32, u32)>,
        shared_perimeters: Vec<f64>,
    ) -> PyResult<()> {
        let metric =
            match (total_perimeter_values, boundary_perimeter_values) {
                (Some(total), None) => {
                    PreparedPolsbyPopper::new(area_values, total, edges, shared_perimeters)
                }
                (None, Some(boundary)) => PreparedPolsbyPopper::from_boundary_perimeters(
                    area_values,
                    boundary,
                    edges,
                    shared_perimeters,
                ),
                _ => return Err(PyValueError::new_err(
                    "provide exactly one of total_perimeter_values or boundary_perimeter_values",
                )),
            };
        self.metrics
            .push(RustBackedMetric::PolsbyPopper(metric.map_err(value_error)?));
        Ok(())
    }

    #[pyo3(signature = (
        rows,
        edges,
        graph_node_count,
        source_crs=None,
        target_crs=None,
        allow_geographic_crs=false,
        allow_unknown_crs=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn add_polsby_popper_geometry(
        &mut self,
        py: Python<'_>,
        rows: Vec<Vec<u8>>,
        edges: Vec<(u32, u32)>,
        graph_node_count: usize,
        source_crs: Option<String>,
        target_crs: Option<String>,
        allow_geographic_crs: bool,
        allow_unknown_crs: bool,
    ) -> PyResult<()> {
        let result = py.detach(|| {
            PolsbyPopperGeometries::from_wkb(
                &rows,
                WkbGeometryLoadOptions {
                    source_crs: source_crs.as_deref(),
                    target_crs: target_crs.as_deref(),
                    allow_geographic_crs,
                    allow_unknown_crs,
                },
                &edges,
                graph_node_count,
            )
            .and_then(|geometry| PreparedPolsbyPopper::from_geometry(edges, geometry))
            .map_err(|error| error.to_string())
        });
        self.metrics.push(RustBackedMetric::PolsbyPopper(
            result.map_err(PyValueError::new_err)?,
        ));
        Ok(())
    }

    fn score_many(
        &self,
        py: Python<'_>,
        assignments: Vec<Vec<u16>>,
    ) -> PyResult<(Vec<Vec<f64>>, Vec<u16>)> {
        if self.metrics.is_empty() {
            return Err(PyRuntimeError::new_err(
                "cannot score without a prepared metric",
            ));
        }

        py.detach(|| {
            self.score_many_inner(&assignments)
                .map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)
    }
}

impl RustBackendScorer {
    fn score_many_inner(
        &self,
        assignments: &[Vec<u16>],
    ) -> crate::error::Result<(Vec<Vec<f64>>, Vec<u16>)> {
        let mut expected_districts = None;
        let mut district_ids = Vec::new();
        let mut rows = Vec::with_capacity(assignments.len());

        for assignment in assignments {
            let outputs = self
                .metrics
                .iter()
                .map(|metric| metric.score(assignment))
                .collect::<crate::error::Result<Vec<_>>>()?;
            let observed = outputs[0].observed;
            if outputs.iter().any(|output| output.observed != observed) {
                return Err(crate::error::invalid_data(
                    "prepared metrics observed different district sets",
                )
                .into());
            }

            match expected_districts {
                Some(expected) => {
                    validate_district_set_unchanged(observed, expected, "score")?;
                }
                None => {
                    expected_districts = Some(observed);
                    district_ids = sorted_district_ids(observed);
                }
            }

            let mut row = Vec::new();
            for output in outputs {
                for table_index in 0..output.table_count {
                    let table = output.table(table_index).ok_or_else(|| {
                        crate::error::invalid_data("prepared metric returned a malformed table")
                    })?;
                    row.extend(
                        district_ids
                            .iter()
                            .map(|&district| table[district as usize]),
                    );
                }
            }
            rows.push(row);
        }

        Ok((rows, district_ids))
    }
}

fn value_error(error: crate::error::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[pymodule]
fn _rust_backend(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustBackendScorer>()?;
    Ok(())
}
