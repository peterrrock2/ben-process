use crate::district::sorted_district_ids;
use crate::error::{invalid_data, Result};
use crate::metrics::PreparedMetricOutput;
use crate::output::parquet::DistrictMetricWriter;
use crate::pipeline::{parquet_compression, PARQUET_BATCH_ROWS};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io;
use std::path::{Component, Path, PathBuf};

pub(crate) struct RunMetricLayout {
    pub(crate) table_paths: Vec<String>,
}

pub(crate) struct RunDirectorySink {
    output_path: PathBuf,
    temp_path: PathBuf,
    writers: Vec<Vec<DistrictMetricWriter>>,
    observed: Option<u128>,
    armed: bool,
}

impl RunDirectorySink {
    pub(crate) fn new(output_dir: &str, layouts: &[RunMetricLayout]) -> Result<Self> {
        let output_path = PathBuf::from(output_dir);
        if output_path.as_os_str().is_empty() {
            return Err(invalid_data("output directory must not be empty").into());
        }
        if output_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("output path {:?} already exists", output_path),
            )
            .into());
        }

        let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let name = output_path
            .file_name()
            .ok_or_else(|| invalid_data("output directory must name a directory"))?
            .to_string_lossy();
        let temp_path = (0..100)
            .find_map(|_| {
                let candidate = parent.join(format!(
                    ".{name}.tmp-{}-{:016x}",
                    std::process::id(),
                    fastrand::u64(..)
                ));
                match fs::create_dir(&candidate) {
                    Ok(()) => Some(Ok(candidate)),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .transpose()?
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::AlreadyExists, "temporary path collision")
            })?;

        let result = Self::writers(&temp_path, layouts).map(|writers| Self {
            output_path,
            temp_path: temp_path.clone(),
            writers,
            observed: None,
            armed: true,
        });
        if result.is_err() {
            let _ = fs::remove_dir_all(temp_path);
        }
        result
    }

    fn writers(
        temp_path: &Path,
        layouts: &[RunMetricLayout],
    ) -> Result<Vec<Vec<DistrictMetricWriter>>> {
        layouts
            .iter()
            .map(|layout| {
                layout
                    .table_paths
                    .iter()
                    .map(|relative| {
                        let relative_path = Path::new(relative);
                        if relative_path.is_absolute()
                            || relative_path
                                .components()
                                .any(|part| !matches!(part, Component::Normal(_)))
                        {
                            return Err(invalid_data(format!(
                                "invalid run-directory table path {relative:?}"
                            ))
                            .into());
                        }
                        let path = temp_path.join(relative_path);
                        fs::create_dir_all(
                            path.parent()
                                .expect("a relative table path should have a parent"),
                        )?;
                        Ok(DistrictMetricWriter::new(
                            Box::new(move || File::create(path)),
                            parquet_compression(false),
                            PARQUET_BATCH_ROWS,
                        ))
                    })
                    .collect()
            })
            .collect()
    }

    pub(crate) fn push(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted: u64,
        outputs: &[PreparedMetricOutput],
    ) -> Result<()> {
        if outputs.len() != self.writers.len() {
            return Err(invalid_data("prepared metric output count changed").into());
        }
        let observed = outputs
            .first()
            .ok_or_else(|| invalid_data("cannot write a row without prepared metrics"))?
            .observed;
        if outputs.iter().any(|output| output.observed != observed) {
            return Err(invalid_data("prepared metrics observed different district sets").into());
        }
        self.observed.get_or_insert(observed);

        for (output, writers) in outputs.iter().zip(&mut self.writers) {
            if output.table_count != writers.len() {
                return Err(
                    invalid_data("prepared metric returned an unexpected table count").into(),
                );
            }
            for (table_index, writer) in writers.iter_mut().enumerate() {
                let table = output
                    .table(table_index)
                    .ok_or_else(|| invalid_data("prepared metric returned a malformed table"))?;
                writer.push_row(step, n_reps, accepted, (observed, table))?;
            }
        }
        Ok(())
    }

    pub(crate) fn finish(mut self, source: &str, metrics: Value) -> Result<()> {
        for writer in std::mem::take(&mut self.writers).into_iter().flatten() {
            writer.finish()?;
        }

        let district_ids = self.observed.map(sorted_district_ids).unwrap_or_default();
        let mut table_schema = vec![
            json!({"name": "step", "dtype": "uint64"}),
            json!({"name": "n_reps", "dtype": "uint32"}),
            json!({"name": "accepted_count", "dtype": "uint64"}),
        ];
        table_schema.extend(district_ids.iter().map(|district_id| {
            json!({
                "name": format!("district_{district_id}"),
                "dtype": "float64",
                "district_id": district_id,
            })
        }));
        let manifest = json!({
            "format_version": 1,
            "source": {"path": source},
            "district_ids": district_ids,
            "table_schema": table_schema,
            "metrics": metrics,
        });
        let manifest_file = File::create(self.temp_path.join("manifest.json"))?;
        serde_json::to_writer_pretty(manifest_file, &manifest)
            .map_err(|error| invalid_data(format!("failed to write manifest: {error}")))?;

        fs::rename(&self.temp_path, &self.output_path)?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for RunDirectorySink {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.temp_path);
        }
    }
}
