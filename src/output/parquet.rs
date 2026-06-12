//! Batched Parquet writers for the per-plan metric modes.
//!
//! All writers create their output file **lazily**: construction takes a file factory and performs
//! no I/O, the file is created when the first row is pushed (i.e. after the first assignment has
//! decoded successfully), and `finish` creates it then if no row ever arrived so a successful
//! zero-frame run still leaves a readable, empty-schema output. A run that fails before the first
//! decoded assignment therefore leaves no output file behind.

use crate::district::sorted_district_ids;
use polars::io::parquet::write::BatchedWriter;
use polars::prelude::*;
use std::fs::File;
use std::io;

/// Deferred output-file constructor. Invoked at most once, on the first pushed row (or at
/// `finish` for a zero-row run). The factory owns any directory creation its path needs.
pub(crate) type FileFactory = Box<dyn FnOnce() -> io::Result<File>>;

pub(crate) struct F64MetricWriter {
    make_file: Option<FileFactory>,
    writer: Option<BatchedWriter<File>>,
    compression: ParquetCompression,
    metric_column_name: String,
    batch_rows: usize,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
    metric_values: Vec<f64>,
}

pub(crate) struct U32KeyedMetricWriter {
    make_file: Option<FileFactory>,
    writer: Option<BatchedWriter<File>>,
    compression: ParquetCompression,
    key_column_name: String,
    metric_column_name: String,
    batch_rows: usize,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
    metric_keys: Vec<String>,
    metric_values: Vec<u32>,
}

/// Streaming writer for per-district metric tables (`step`, `n_reps`, `accepted_count`, plus one
/// `district_N` column per observed district).
///
/// The district-column schema is fixed from the **first pushed row's** observed set; callers never
/// pass district ids. This is sound because `run_pipeline` enforces a fixed district set across
/// the ensemble, so the first row's set is every row's set.
pub(crate) struct DistrictMetricWriter {
    make_file: Option<FileFactory>,
    writer: Option<BatchedWriter<File>>,
    compression: ParquetCompression,
    district_ids: Vec<u16>,
    batch_rows: usize,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
    district_columns: Vec<Vec<Option<f64>>>,
}

impl F64MetricWriter {
    pub(crate) fn new(
        make_file: FileFactory,
        metric_column_name: impl Into<String>,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> Self {
        Self {
            make_file: Some(make_file),
            writer: None,
            compression,
            metric_column_name: metric_column_name.into(),
            batch_rows,
            sample_numbers: Vec::with_capacity(batch_rows),
            n_reps_numbers: Vec::with_capacity(batch_rows),
            accepted_numbers: Vec::with_capacity(batch_rows),
            metric_values: Vec::with_capacity(batch_rows),
        }
    }

    fn create_writer(&mut self) -> crate::error::Result<()> {
        let make_file = self
            .make_file
            .take()
            .expect("output file factory should be unconsumed before the writer exists");
        let file = make_file()?;
        let empty_df = empty_f64_metric_df(&self.metric_column_name)?;
        self.writer = Some(
            ParquetWriter::new(file)
                .with_compression(self.compression)
                .batched(empty_df.schema())?,
        );
        Ok(())
    }

    pub(crate) fn push(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted_count: u64,
        value: f64,
    ) -> crate::error::Result<()> {
        if self.writer.is_none() {
            self.create_writer()?;
        }

        self.sample_numbers.push(step);
        self.n_reps_numbers.push(n_reps);
        self.accepted_numbers.push(accepted_count);
        self.metric_values.push(value);

        if self.sample_numbers.len() >= self.batch_rows {
            self.flush()?;
        }

        Ok(())
    }

    pub(crate) fn finish(mut self) -> crate::error::Result<()> {
        if self.writer.is_none() {
            // Zero rows pushed but the run completed: emit the empty-schema output now.
            self.create_writer()?;
        }
        self.flush()?;
        self.writer
            .take()
            .expect("writer should exist after create_writer")
            .finish()?;
        Ok(())
    }

    fn flush(&mut self) -> crate::error::Result<()> {
        if self.sample_numbers.is_empty() {
            return Ok(());
        }

        let df = f64_metric_batch_to_df(
            &self.metric_column_name,
            &mut self.sample_numbers,
            &mut self.n_reps_numbers,
            &mut self.accepted_numbers,
            &mut self.metric_values,
        )?;
        self.writer
            .as_mut()
            .expect("writer should exist once rows are buffered")
            .write_batch(&df)?;
        self.reset_buffers();

        Ok(())
    }

    fn reset_buffers(&mut self) {
        self.sample_numbers = Vec::with_capacity(self.batch_rows);
        self.n_reps_numbers = Vec::with_capacity(self.batch_rows);
        self.accepted_numbers = Vec::with_capacity(self.batch_rows);
        self.metric_values = Vec::with_capacity(self.batch_rows);
    }
}

impl U32KeyedMetricWriter {
    pub(crate) fn new(
        make_file: FileFactory,
        key_column_name: impl Into<String>,
        metric_column_name: impl Into<String>,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> Self {
        Self {
            make_file: Some(make_file),
            writer: None,
            compression,
            key_column_name: key_column_name.into(),
            metric_column_name: metric_column_name.into(),
            batch_rows,
            sample_numbers: Vec::with_capacity(batch_rows),
            n_reps_numbers: Vec::with_capacity(batch_rows),
            accepted_numbers: Vec::with_capacity(batch_rows),
            metric_keys: Vec::with_capacity(batch_rows),
            metric_values: Vec::with_capacity(batch_rows),
        }
    }

    fn create_writer(&mut self) -> crate::error::Result<()> {
        let make_file = self
            .make_file
            .take()
            .expect("output file factory should be unconsumed before the writer exists");
        let file = make_file()?;
        let empty_df = empty_u32_keyed_metric_df(&self.key_column_name, &self.metric_column_name)?;
        self.writer = Some(
            ParquetWriter::new(file)
                .with_compression(self.compression)
                .batched(empty_df.schema())?,
        );
        Ok(())
    }

    pub(crate) fn push(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted_count: u64,
        key: impl Into<String>,
        value: u32,
    ) -> crate::error::Result<()> {
        if self.writer.is_none() {
            self.create_writer()?;
        }

        self.sample_numbers.push(step);
        self.n_reps_numbers.push(n_reps);
        self.accepted_numbers.push(accepted_count);
        self.metric_keys.push(key.into());
        self.metric_values.push(value);

        if self.sample_numbers.len() >= self.batch_rows {
            self.flush()?;
        }

        Ok(())
    }

    pub(crate) fn finish(mut self) -> crate::error::Result<()> {
        if self.writer.is_none() {
            self.create_writer()?;
        }
        self.flush()?;
        self.writer
            .take()
            .expect("writer should exist after create_writer")
            .finish()?;
        Ok(())
    }

    fn flush(&mut self) -> crate::error::Result<()> {
        if self.sample_numbers.is_empty() {
            return Ok(());
        }

        let df = u32_keyed_metric_batch_to_df(
            &self.key_column_name,
            &self.metric_column_name,
            &mut self.sample_numbers,
            &mut self.n_reps_numbers,
            &mut self.accepted_numbers,
            &mut self.metric_keys,
            &mut self.metric_values,
        )?;
        self.writer
            .as_mut()
            .expect("writer should exist once rows are buffered")
            .write_batch(&df)?;
        self.reset_buffers();

        Ok(())
    }

    fn reset_buffers(&mut self) {
        self.sample_numbers = Vec::with_capacity(self.batch_rows);
        self.n_reps_numbers = Vec::with_capacity(self.batch_rows);
        self.accepted_numbers = Vec::with_capacity(self.batch_rows);
        self.metric_keys = Vec::with_capacity(self.batch_rows);
        self.metric_values = Vec::with_capacity(self.batch_rows);
    }
}

impl DistrictMetricWriter {
    pub(crate) fn new(
        make_file: FileFactory,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> Self {
        Self {
            make_file: Some(make_file),
            writer: None,
            compression,
            district_ids: Vec::new(),
            batch_rows,
            sample_numbers: Vec::with_capacity(batch_rows),
            n_reps_numbers: Vec::with_capacity(batch_rows),
            accepted_numbers: Vec::with_capacity(batch_rows),
            district_columns: Vec::new(),
        }
    }

    fn create_writer(&mut self) -> crate::error::Result<()> {
        let make_file = self
            .make_file
            .take()
            .expect("output file factory should be unconsumed before the writer exists");
        let file = make_file()?;
        let empty_df = empty_district_metric_df(&self.district_ids)?;
        self.writer = Some(
            ParquetWriter::new(file)
                .with_compression(self.compression)
                .batched(empty_df.schema())?,
        );
        Ok(())
    }

    /// Push one plan's row. `values` is indexed by district label (`values[d]` is district `d`'s
    /// value, length `max observed label + 1`); the writer selects the observed labels itself.
    ///
    /// The first call fixes the schema from `observed` and creates the output file.
    pub(crate) fn push_row(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted_count: u64,
        observed: u128,
        values: &[f64],
    ) -> crate::error::Result<()> {
        if self.writer.is_none() {
            self.district_ids = sorted_district_ids(observed);
            self.district_columns = self
                .district_ids
                .iter()
                .map(|_| Vec::with_capacity(self.batch_rows))
                .collect();
            self.create_writer()?;
        }

        self.sample_numbers.push(step);
        self.n_reps_numbers.push(n_reps);
        self.accepted_numbers.push(accepted_count);

        for (column_index, &district_id) in self.district_ids.iter().enumerate() {
            self.district_columns[column_index].push(values.get(district_id as usize).copied());
        }

        if self.sample_numbers.len() >= self.batch_rows {
            self.flush()?;
        }

        Ok(())
    }

    pub(crate) fn finish(mut self) -> crate::error::Result<()> {
        if self.writer.is_none() {
            // Zero rows pushed but the run completed: emit the empty-schema output (no district
            // columns) now.
            self.create_writer()?;
        }
        self.flush()?;
        self.writer
            .take()
            .expect("writer should exist after create_writer")
            .finish()?;
        Ok(())
    }

    fn flush(&mut self) -> crate::error::Result<()> {
        if self.sample_numbers.is_empty() {
            return Ok(());
        }

        let df = district_metric_batch_to_df(
            &self.district_ids,
            &mut self.sample_numbers,
            &mut self.n_reps_numbers,
            &mut self.accepted_numbers,
            &mut self.district_columns,
        )?;
        self.writer
            .as_mut()
            .expect("writer should exist once rows are buffered")
            .write_batch(&df)?;
        self.reset_buffers();

        Ok(())
    }

    fn reset_buffers(&mut self) {
        self.sample_numbers = Vec::with_capacity(self.batch_rows);
        self.n_reps_numbers = Vec::with_capacity(self.batch_rows);
        self.accepted_numbers = Vec::with_capacity(self.batch_rows);
        self.district_columns = self
            .district_ids
            .iter()
            .map(|_| Vec::with_capacity(self.batch_rows))
            .collect();
    }
}

fn empty_f64_metric_df(metric_column_name: &str) -> PolarsResult<DataFrame> {
    DataFrame::new_infer_height(vec![
        Series::new("step".into(), Vec::<u64>::new()).into(),
        Series::new("n_reps".into(), Vec::<u32>::new()).into(),
        Series::new("accepted_count".into(), Vec::<u64>::new()).into(),
        Series::new(metric_column_name.into(), Vec::<f64>::new()).into(),
    ])
}

fn empty_district_metric_df(district_ids: &[u16]) -> PolarsResult<DataFrame> {
    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), Vec::<u64>::new()).into(),
        Series::new("n_reps".into(), Vec::<u32>::new()).into(),
        Series::new("accepted_count".into(), Vec::<u64>::new()).into(),
    ])?;

    for &district_id in district_ids {
        df.with_column(
            Series::new(
                format!("district_{}", district_id).into(),
                Vec::<Option<f64>>::new(),
            )
            .into(),
        )?;
    }

    Ok(df)
}

fn f64_metric_batch_to_df(
    metric_column_name: &str,
    sample_numbers: &mut Vec<u64>,
    n_reps_numbers: &mut Vec<u32>,
    accepted_numbers: &mut Vec<u64>,
    metric_values: &mut Vec<f64>,
) -> PolarsResult<DataFrame> {
    DataFrame::new_infer_height(vec![
        Series::new("step".into(), std::mem::take(sample_numbers)).into(),
        Series::new("n_reps".into(), std::mem::take(n_reps_numbers)).into(),
        Series::new("accepted_count".into(), std::mem::take(accepted_numbers)).into(),
        Series::new(metric_column_name.into(), std::mem::take(metric_values)).into(),
    ])
}

fn empty_u32_keyed_metric_df(
    key_column_name: &str,
    metric_column_name: &str,
) -> PolarsResult<DataFrame> {
    DataFrame::new_infer_height(vec![
        Series::new("step".into(), Vec::<u64>::new()).into(),
        Series::new("n_reps".into(), Vec::<u32>::new()).into(),
        Series::new("accepted_count".into(), Vec::<u64>::new()).into(),
        Series::new(key_column_name.into(), Vec::<String>::new()).into(),
        Series::new(metric_column_name.into(), Vec::<u32>::new()).into(),
    ])
}

fn u32_keyed_metric_batch_to_df(
    key_column_name: &str,
    metric_column_name: &str,
    sample_numbers: &mut Vec<u64>,
    n_reps_numbers: &mut Vec<u32>,
    accepted_numbers: &mut Vec<u64>,
    metric_keys: &mut Vec<String>,
    metric_values: &mut Vec<u32>,
) -> PolarsResult<DataFrame> {
    DataFrame::new_infer_height(vec![
        Series::new("step".into(), std::mem::take(sample_numbers)).into(),
        Series::new("n_reps".into(), std::mem::take(n_reps_numbers)).into(),
        Series::new("accepted_count".into(), std::mem::take(accepted_numbers)).into(),
        Series::new(key_column_name.into(), std::mem::take(metric_keys)).into(),
        Series::new(metric_column_name.into(), std::mem::take(metric_values)).into(),
    ])
}

fn district_metric_batch_to_df(
    district_ids: &[u16],
    sample_numbers: &mut Vec<u64>,
    n_reps_numbers: &mut Vec<u32>,
    accepted_numbers: &mut Vec<u64>,
    district_columns: &mut [Vec<Option<f64>>],
) -> PolarsResult<DataFrame> {
    let mut df = DataFrame::new_infer_height(vec![
        Series::new("step".into(), std::mem::take(sample_numbers)).into(),
        Series::new("n_reps".into(), std::mem::take(n_reps_numbers)).into(),
        Series::new("accepted_count".into(), std::mem::take(accepted_numbers)).into(),
    ])?;

    for (column_index, &district_id) in district_ids.iter().enumerate() {
        let column = std::mem::take(&mut district_columns[column_index]);
        df.with_column(Series::new(format!("district_{}", district_id).into(), column).into())?;
    }

    Ok(df)
}

/// Write the per-node changed-assignment counts as a two-column Parquet table (`node`,
/// `changed_assignments`), one row per graph node. The total accepted-frame count is encoded in the
/// output filename.
pub(crate) fn write_changed_assignments(
    file: File,
    changed_assignments: &[f64],
    compression: ParquetCompression,
) -> crate::error::Result<()> {
    let nodes: Vec<u32> = (0..changed_assignments.len() as u32).collect();
    let mut df = DataFrame::new_infer_height(vec![
        Series::new("node".into(), nodes).into(),
        Series::new("changed_assignments".into(), changed_assignments.to_vec()).into(),
    ])?;
    ParquetWriter::new(file)
        .with_compression(compression)
        .finish(&mut df)?;
    Ok(())
}

/// Write the unique-plan counts as a single-row Parquet table (`unique_plans`,
/// `total_accepted_frames`).
pub(crate) fn write_unique_plans(
    file: File,
    unique_plans: u64,
    total_accepted_frames: u64,
    compression: ParquetCompression,
) -> crate::error::Result<()> {
    let mut df = DataFrame::new_infer_height(vec![
        Series::new("unique_plans".into(), vec![unique_plans]).into(),
        Series::new("total_accepted_frames".into(), vec![total_accepted_frames]).into(),
    ])?;
    ParquetWriter::new(file)
        .with_compression(compression)
        .finish(&mut df)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::DistrictMetricWriter;
    use crate::pipeline::parquet_compression;
    use polars::prelude::{ParquetReader, SerReader};
    use std::fs::File;
    use tempfile::tempdir;

    fn writer_for(path: std::path::PathBuf, batch_rows: usize) -> DistrictMetricWriter {
        DistrictMetricWriter::new(
            Box::new(move || File::create(path)),
            parquet_compression(false),
            batch_rows,
        )
    }

    #[test]
    fn district_metric_writer_fixes_schema_from_first_row_and_appends_batches() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("out.parquet");
        let mut writer = writer_for(path.clone(), 2);

        // No I/O happens at construction: the file must not exist until the first row arrives.
        assert!(!path.exists(), "no output file before the first pushed row");

        let observed = (1u128 << 1) | (1u128 << 2);
        // values[d] is district d's value; district 0 is unobserved and must not get a column.
        writer
            .push_row(1, 1, 1, observed, &[9.9, 0.1, 0.3])
            .unwrap();
        assert!(path.exists(), "first pushed row creates the output file");
        writer
            .push_row(2, 1, 2, observed, &[9.9, 0.2, 0.4])
            .unwrap();
        writer.push_row(3, 2, 3, observed, &[9.9, 0.5]).unwrap();
        writer.finish().unwrap();

        let df = ParquetReader::new(&mut File::open(&path).unwrap())
            .finish()
            .unwrap();
        assert!(
            df.column("district_0").is_err(),
            "unobserved district 0 must not appear in the schema"
        );
        assert_eq!(
            df.column("step")
                .unwrap()
                .u64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            df.column("district_1")
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![Some(0.1), Some(0.2), Some(0.5)]
        );
        // The third row's values slice is too short for district 2 → a null cell, not a panic.
        assert_eq!(
            df.column("district_2")
                .unwrap()
                .f64()
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![Some(0.3), Some(0.4), None]
        );
    }

    #[test]
    fn district_metric_writer_emits_empty_schema_file_when_no_rows_arrive() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.parquet");
        let writer = writer_for(path.clone(), 2);

        assert!(!path.exists());
        writer.finish().unwrap();

        let df = ParquetReader::new(&mut File::open(&path).unwrap())
            .finish()
            .unwrap();
        assert_eq!(df.height(), 0);
        assert_eq!(
            df.get_column_names(),
            vec!["step", "n_reps", "accepted_count"]
        );
    }
}
