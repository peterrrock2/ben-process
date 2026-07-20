use crate::district::{sorted_district_ids, validate_district_set_unchanged};
use crate::geometry::{PolsbyPopperGeometries, ReockGeometries, WkbGeometryLoadOptions};
use crate::metrics::polsby_popper::{IncrementalPolsbyPopper, PreparedPolsbyPopper};
use crate::metrics::reock::{IncrementalReock, PreparedReock};
use crate::metrics::tally_keys::{IncrementalTallies, PreparedTally};
use crate::metrics::twodelta::{
    run_incremental_twodelta, DeltaChange, IncrementalTwoDeltaMetric, TwoDeltaRow,
    TwoDeltaRunOptions,
};
use crate::metrics::PreparedMetricOutput;
use crate::output::run_directory::{RunDirectorySink, RunMetricLayout};
use crate::pipeline::{run_pipeline, AssignmentLengthCheck};
use ben::BenVariant;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::Value;

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

    fn node_count(&self) -> usize {
        match self {
            Self::Tally(metric) => metric.node_count(),
            Self::Reock(metric) => metric.node_count(),
            Self::PolsbyPopper(metric) => metric.node_count(),
        }
    }

    fn node_count_label(&self) -> &'static str {
        match self {
            Self::Reock(_) => "geometry row count",
            Self::Tally(_) | Self::PolsbyPopper(_) => "graph node count",
        }
    }

    fn metric_key(&self) -> &'static str {
        match self {
            Self::Tally(_) => "tally",
            Self::Reock(_) => "reock",
            Self::PolsbyPopper(_) => "polsby_popper",
        }
    }

    fn layout(&self) -> RunMetricLayout {
        let table_paths = match self {
            Self::Tally(metric) => (0..metric.table_count())
                .map(|index| format!("tally/{index:04}.parquet"))
                .collect(),
            Self::Reock(_) => vec!["reock/scores.parquet".to_string()],
            Self::PolsbyPopper(_) => vec!["polsby_popper/scores.parquet".to_string()],
        };
        RunMetricLayout { table_paths }
    }
}

enum IncrementalRustBackedMetric<'a> {
    Tally(IncrementalTallies<'a>),
    Reock(IncrementalReock<'a>),
    PolsbyPopper(IncrementalPolsbyPopper<'a>),
}

impl<'a> IncrementalRustBackedMetric<'a> {
    fn new(metric: &'a RustBackedMetric) -> Self {
        match metric {
            RustBackedMetric::Tally(metric) => Self::Tally(IncrementalTallies::new(metric)),
            RustBackedMetric::Reock(metric) => Self::Reock(IncrementalReock::new(metric)),
            RustBackedMetric::PolsbyPopper(metric) => {
                Self::PolsbyPopper(IncrementalPolsbyPopper::new(metric))
            }
        }
    }

    fn output(&self) -> crate::error::Result<PreparedMetricOutput> {
        match self {
            Self::Tally(metric) => Ok(metric.output()),
            Self::Reock(metric) => Ok(metric.output()),
            Self::PolsbyPopper(metric) => metric.output(),
        }
    }
}

impl IncrementalTwoDeltaMetric for IncrementalRustBackedMetric<'_> {
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        match self {
            Self::Tally(metric) => metric.seed(assignment),
            Self::Reock(metric) => metric.seed(assignment),
            Self::PolsbyPopper(metric) => metric.seed(assignment),
        }
    }

    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[DeltaChange],
    ) -> crate::error::Result<()> {
        match self {
            Self::Tally(metric) => metric.update_delta(before, changes),
            Self::Reock(metric) => metric.update_delta(before, changes),
            Self::PolsbyPopper(metric) => metric.update_delta(before, changes),
        }
    }

    fn observed(&self) -> u128 {
        match self {
            Self::Tally(metric) => metric.observed(),
            Self::Reock(metric) => metric.observed(),
            Self::PolsbyPopper(metric) => metric.observed(),
        }
    }
}

struct CompositeTwoDeltaMetric<'a> {
    metrics: Vec<IncrementalRustBackedMetric<'a>>,
}

impl CompositeTwoDeltaMetric<'_> {
    fn outputs(&self) -> crate::error::Result<Vec<PreparedMetricOutput>> {
        let outputs = self
            .metrics
            .iter()
            .map(IncrementalRustBackedMetric::output)
            .collect::<crate::error::Result<Vec<_>>>()?;
        validate_matching_observed(&outputs)?;
        Ok(outputs)
    }

    fn validate_observed(&self) -> crate::error::Result<()> {
        let Some(first) = self.metrics.first() else {
            return Err(
                crate::error::invalid_data("cannot score without a prepared metric").into(),
            );
        };
        if self
            .metrics
            .iter()
            .any(|metric| metric.observed() != first.observed())
        {
            return Err(crate::error::invalid_data(
                "prepared metrics observed different district sets",
            )
            .into());
        }
        Ok(())
    }
}

impl IncrementalTwoDeltaMetric for CompositeTwoDeltaMetric<'_> {
    fn seed(&mut self, assignment: &[u16]) -> crate::error::Result<()> {
        for metric in &mut self.metrics {
            metric.seed(assignment)?;
        }
        self.validate_observed()
    }

    fn update_delta(
        &mut self,
        before: &[u16],
        changes: &[DeltaChange],
    ) -> crate::error::Result<()> {
        for metric in &mut self.metrics {
            metric.update_delta(before, changes)?;
        }
        self.validate_observed()
    }

    fn observed(&self) -> u128 {
        self.metrics
            .first()
            .map_or(0, IncrementalTwoDeltaMetric::observed)
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

    fn score_ben_file(
        &self,
        py: Python<'_>,
        output_dir: String,
        source: String,
        metrics_json: String,
    ) -> PyResult<()> {
        if self.metrics.is_empty() {
            return Err(PyRuntimeError::new_err(
                "cannot score without a prepared metric",
            ));
        }

        py.detach(|| {
            self.score_ben_file_inner(&output_dir, &source, &metrics_json)
                .map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)
    }
}

impl RustBackendScorer {
    fn score_outputs(
        &self,
        assignment: &[u16],
    ) -> crate::error::Result<(u128, Vec<PreparedMetricOutput>)> {
        let outputs = self
            .metrics
            .iter()
            .map(|metric| metric.score(assignment))
            .collect::<crate::error::Result<Vec<_>>>()?;
        let observed = validate_matching_observed(&outputs)?;
        Ok((observed, outputs))
    }

    fn score_many_inner(
        &self,
        assignments: &[Vec<u16>],
    ) -> crate::error::Result<(Vec<Vec<f64>>, Vec<u16>)> {
        let mut expected_districts = None;
        let mut district_ids = Vec::new();
        let mut rows = Vec::with_capacity(assignments.len());

        for assignment in assignments {
            let (observed, outputs) = self.score_outputs(assignment)?;

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

    fn score_ben_file_inner(
        &self,
        output_dir: &str,
        source_path: &str,
        metrics_json: &str,
    ) -> crate::error::Result<()> {
        let layouts: Vec<_> = self.metrics.iter().map(RustBackedMetric::layout).collect();
        let manifest_metrics: Value = serde_json::from_str(metrics_json).map_err(|error| {
            crate::error::invalid_data(format!("invalid run-directory metric metadata: {error}"))
        })?;
        validate_manifest_metrics(&manifest_metrics, &self.metrics, &layouts)?;

        let mut sink = RunDirectorySink::new(output_dir, &layouts)?;
        let resolved = crate::input::resolve(source_path)?;
        let source = resolved.source;
        let first_metric = self
            .metrics
            .first()
            .ok_or_else(|| crate::error::invalid_data("cannot score without a prepared metric"))?;
        let expected_len = first_metric.node_count();
        let expected_len_label = first_metric.node_count_label();

        if source.variant()? == BenVariant::TwoDelta {
            let mut composite = CompositeTwoDeltaMetric {
                metrics: self
                    .metrics
                    .iter()
                    .map(IncrementalRustBackedMetric::new)
                    .collect(),
            };
            run_incremental_twodelta(
                &source,
                TwoDeltaRunOptions {
                    expected_len,
                    expected_len_label,
                    output_name: "score",
                    show_progress: false,
                    max_samples: None,
                },
                &mut composite,
                |metric,
                 TwoDeltaRow {
                     step,
                     n_reps,
                     accepted,
                 }| { sink.push(step, n_reps, accepted, &metric.outputs()?) },
            )?;
        } else {
            run_pipeline(
                &source,
                AssignmentLengthCheck::Exact {
                    expected: expected_len,
                    label: expected_len_label,
                },
                "score",
                |assignment, _n_reps| self.score_outputs(assignment),
                |step, n_reps, accepted, outputs| sink.push(step, n_reps, accepted, &outputs),
                false,
                None,
            )?;
        }

        sink.finish(source_path, manifest_metrics)
    }
}

fn validate_matching_observed(outputs: &[PreparedMetricOutput]) -> crate::error::Result<u128> {
    let observed = outputs
        .first()
        .ok_or_else(|| crate::error::invalid_data("cannot score without a prepared metric"))?
        .observed;
    if outputs.iter().any(|output| output.observed != observed) {
        return Err(crate::error::invalid_data(
            "prepared metrics observed different district sets",
        )
        .into());
    }
    Ok(observed)
}

fn validate_manifest_metrics(
    value: &Value,
    metrics: &[RustBackedMetric],
    layouts: &[RunMetricLayout],
) -> crate::error::Result<()> {
    let entries = value
        .as_array()
        .ok_or_else(|| crate::error::invalid_data("manifest metrics must be a JSON array"))?;
    if entries.len() != metrics.len() {
        return Err(crate::error::invalid_data(
            "manifest metric count does not match prepared metric count",
        )
        .into());
    }

    for ((entry, metric), layout) in entries.iter().zip(metrics).zip(layouts) {
        if entry.get("metric_key").and_then(Value::as_str) != Some(metric.metric_key()) {
            return Err(crate::error::invalid_data(
                "manifest metric order does not match prepared metric order",
            )
            .into());
        }
        let tables = entry
            .get("tables")
            .and_then(Value::as_array)
            .ok_or_else(|| crate::error::invalid_data("manifest metric tables must be an array"))?;
        let paths: Option<Vec<_>> = tables
            .iter()
            .map(|table| table.get("path").and_then(Value::as_str))
            .collect();
        if paths.as_deref()
            != Some(
                &layout
                    .table_paths
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>(),
            )
        {
            return Err(crate::error::invalid_data(
                "manifest table paths do not match the run-directory layout",
            )
            .into());
        }
    }
    Ok(())
}

fn value_error(error: crate::error::Error) -> PyErr {
    PyValueError::new_err(error.to_string())
}

type PyBendlAssets = (Option<Py<PyBytes>>, Vec<String>, Option<Py<PyBytes>>);

#[pyfunction]
#[pyo3(signature = (path, asset_name=None))]
fn load_bendl_assets(
    py: Python<'_>,
    path: String,
    asset_name: Option<String>,
) -> PyResult<PyBendlAssets> {
    let assets = py
        .detach(|| {
            crate::input::load_bendl_assets(&path, asset_name.as_deref())
                .map_err(|error| error.to_string())
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        assets
            .embedded_graph
            .map(|bytes| PyBytes::new(py, &bytes).unbind()),
        assets.custom_asset_names,
        assets
            .selected_asset
            .map(|bytes| PyBytes::new(py, &bytes).unbind()),
    ))
}

#[pymodule]
fn _rust_backend(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustBackendScorer>()?;
    module.add_function(wrap_pyfunction!(load_bendl_assets, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        PreparedPolsbyPopper, PreparedReock, PreparedTally, RustBackedMetric, RustBackendScorer,
    };
    use ben::io::bundle::format::AssignmentFormat;
    use ben::io::bundle::BendlWriter;
    use ben::io::writer::BenStreamWriter;
    use ben::BenVariant;
    use geo::Coord;
    use polars::prelude::{ParquetReader, SerReader};
    use serde_json::json;
    use std::fs::File;
    use std::path::Path;
    use tempfile::tempdir;

    fn write_ben(path: &Path, variant: BenVariant, plans: &[Vec<u16>]) {
        let mut writer = BenStreamWriter::for_ben(File::create(path).unwrap(), variant).unwrap();
        for plan in plans {
            writer.write_assignment(plan.clone()).unwrap();
        }
        writer.finish().unwrap();
    }

    fn write_bendl(path: &Path, plans: &[Vec<u16>]) {
        let writer = BendlWriter::new(File::create(path).unwrap(), AssignmentFormat::Ben).unwrap();
        let mut session = writer.into_stream_session().unwrap();
        {
            let mut stream = BenStreamWriter::for_ben(&mut session, BenVariant::Standard).unwrap();
            for plan in plans {
                stream.write_assignment(plan.clone()).unwrap();
            }
            stream.finish().unwrap();
        }
        session
            .finish_into_writer(plans.len() as i64)
            .finish()
            .unwrap();
    }

    fn test_scorer() -> RustBackendScorer {
        let reock_units = (0..4)
            .map(|index| {
                let x = index as f64 * 2.0;
                crate::geometry::ReockUnit {
                    area: 1.0,
                    convex_hull_points: vec![
                        Coord { x, y: 0.0 },
                        Coord { x: x + 1.0, y: 0.0 },
                        Coord { x: x + 1.0, y: 1.0 },
                        Coord { x, y: 1.0 },
                    ],
                }
            })
            .collect();
        RustBackendScorer {
            metrics: vec![
                RustBackedMetric::Tally(
                    PreparedTally::new(vec![vec![1.0, 2.0, 3.0, 4.0]]).unwrap(),
                ),
                RustBackedMetric::Reock(PreparedReock::new(crate::geometry::ReockGeometries {
                    units: reock_units,
                })),
                RustBackedMetric::PolsbyPopper(
                    PreparedPolsbyPopper::new(
                        vec![1.0; 4],
                        vec![4.0; 4],
                        vec![(0, 1), (1, 2), (2, 3)],
                        vec![1.0; 3],
                    )
                    .unwrap(),
                ),
            ],
        }
    }

    fn manifest_metrics() -> String {
        json!([
            {
                "metric_key": "tally",
                "output_slug": "tally",
                "options": {"keys": ["POP"]},
                "tables": [{"subkey": "POP", "path": "tally/0000.parquet"}],
            },
            {
                "metric_key": "reock",
                "output_slug": "reock",
                "options": {},
                "tables": [{"subkey": null, "path": "reock/scores.parquet"}],
            },
            {
                "metric_key": "polsby_popper",
                "output_slug": "polsby_popper",
                "options": {"source": "geometry"},
                "tables": [{
                    "subkey": null,
                    "path": "polsby_popper/scores.parquet",
                }],
            },
        ])
        .to_string()
    }

    #[test]
    fn run_directory_writes_standard_mkvchain_and_twodelta() {
        let plans = vec![vec![1, 1, 2, 2], vec![1, 1, 2, 2], vec![1, 2, 2, 2]];
        let direct_rows = test_scorer()
            .score_many_inner(&[plans[0].clone(), plans[2].clone()])
            .unwrap()
            .0;
        for variant in [
            BenVariant::Standard,
            BenVariant::MkvChain,
            BenVariant::TwoDelta,
        ] {
            let dir = tempdir().unwrap();
            let source = dir.path().join("plans.ben");
            let output = dir.path().join("scores");
            write_ben(&source, variant, &plans);

            test_scorer()
                .score_ben_file_inner(
                    output.to_str().unwrap(),
                    source.to_str().unwrap(),
                    &manifest_metrics(),
                )
                .unwrap();

            let manifest: serde_json::Value =
                serde_json::from_reader(File::open(output.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(manifest["format_version"], 1);
            assert_eq!(manifest["source"]["path"], source.to_str().unwrap());
            assert_eq!(manifest["district_ids"], json!([1, 2]));
            assert_eq!(
                manifest["table_schema"],
                json!([
                    {"name": "step", "dtype": "uint64"},
                    {"name": "n_reps", "dtype": "uint32"},
                    {"name": "accepted_count", "dtype": "uint64"},
                    {"name": "district_1", "dtype": "float64", "district_id": 1},
                    {"name": "district_2", "dtype": "float64", "district_id": 2},
                ])
            );
            assert_eq!(manifest["metrics"][0]["options"]["keys"], json!(["POP"]));
            assert!(output.join("reock/scores.parquet").exists());
            assert!(output.join("polsby_popper/scores.parquet").exists());

            let table = ParquetReader::new(File::open(output.join("tally/0000.parquet")).unwrap())
                .finish()
                .unwrap();
            assert_eq!(
                table.height(),
                if variant == BenVariant::Standard {
                    3
                } else {
                    2
                }
            );
            assert_eq!(
                table.get_column_names(),
                [
                    "step",
                    "n_reps",
                    "accepted_count",
                    "district_1",
                    "district_2",
                ]
            );

            let expected_plan_rows = if variant == BenVariant::Standard {
                vec![0, 0, 1]
            } else {
                vec![0, 1]
            };
            for (path, offset) in [
                ("tally/0000.parquet", 0),
                ("reock/scores.parquet", 2),
                ("polsby_popper/scores.parquet", 4),
            ] {
                let table = ParquetReader::new(File::open(output.join(path)).unwrap())
                    .finish()
                    .unwrap();
                assert_eq!(table.height(), expected_plan_rows.len());
                let district_1 = table
                    .column("district_1")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_no_null_iter();
                let district_2 = table
                    .column("district_2")
                    .unwrap()
                    .f64()
                    .unwrap()
                    .into_no_null_iter();
                for ((actual_1, actual_2), expected_row) in district_1
                    .zip(district_2)
                    .zip(expected_plan_rows.iter().map(|&index| &direct_rows[index]))
                {
                    assert!((actual_1 - expected_row[offset]).abs() < 1e-12);
                    assert!((actual_2 - expected_row[offset + 1]).abs() < 1e-12);
                }
            }

            let expected_steps = if variant == BenVariant::Standard {
                vec![1, 2, 3]
            } else {
                vec![1, 3]
            };
            let expected_reps = if variant == BenVariant::Standard {
                vec![1, 1, 1]
            } else {
                vec![2, 1]
            };
            assert_eq!(
                table
                    .column("step")
                    .unwrap()
                    .u64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect::<Vec<_>>(),
                expected_steps
            );
            assert_eq!(
                table
                    .column("n_reps")
                    .unwrap()
                    .u32()
                    .unwrap()
                    .into_no_null_iter()
                    .collect::<Vec<_>>(),
                expected_reps
            );
            assert_eq!(
                table
                    .column("accepted_count")
                    .unwrap()
                    .u64()
                    .unwrap()
                    .into_no_null_iter()
                    .collect::<Vec<_>>(),
                (1..=expected_plan_rows.len() as u64).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn score_ben_file_accepts_bendl_input() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("plans.bendl");
        let output = dir.path().join("scores");
        write_bendl(&source, &[vec![1, 1, 2, 2]]);

        test_scorer()
            .score_ben_file_inner(
                output.to_str().unwrap(),
                source.to_str().unwrap(),
                &manifest_metrics(),
            )
            .unwrap();

        assert!(output.join("manifest.json").exists());
        assert!(output.join("tally/0000.parquet").exists());
    }

    #[test]
    fn run_directory_handles_empty_input_and_never_overwrites() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("empty.ben");
        let output = dir.path().join("scores");
        write_ben(&source, BenVariant::Standard, &[]);

        let scorer = test_scorer();
        scorer
            .score_ben_file_inner(
                output.to_str().unwrap(),
                source.to_str().unwrap(),
                &manifest_metrics(),
            )
            .unwrap();
        let table = ParquetReader::new(File::open(output.join("tally/0000.parquet")).unwrap())
            .finish()
            .unwrap();
        assert_eq!(table.height(), 0);
        assert_eq!(
            table.get_column_names(),
            ["step", "n_reps", "accepted_count"]
        );
        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(output.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["district_ids"], json!([]));
        assert_eq!(manifest["table_schema"].as_array().unwrap().len(), 3);

        let error = scorer
            .score_ben_file_inner(
                output.to_str().unwrap(),
                source.to_str().unwrap(),
                &manifest_metrics(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("already exists"));
        assert!(output.join("manifest.json").exists());
    }

    #[test]
    fn failed_run_removes_temporary_directory() {
        let dir = tempdir().unwrap();
        let output = dir.path().join("scores");
        let missing = dir.path().join("missing.ben");

        test_scorer()
            .score_ben_file_inner(
                output.to_str().unwrap(),
                missing.to_str().unwrap(),
                &manifest_metrics(),
            )
            .unwrap_err();

        assert!(!output.exists());
        assert!(std::fs::read_dir(dir.path()).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".scores.tmp-")));
    }
}
