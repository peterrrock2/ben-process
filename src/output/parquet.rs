use polars::io::parquet::write::BatchedWriter;
use polars::prelude::*;
use std::fs::File;

pub(crate) struct F64MetricWriter {
    writer: BatchedWriter<File>,
    metric_column_name: String,
    batch_rows: usize,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
    metric_values: Vec<f64>,
}

pub(crate) struct U32KeyedMetricWriter {
    writer: BatchedWriter<File>,
    key_column_name: String,
    metric_column_name: String,
    batch_rows: usize,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
    metric_keys: Vec<String>,
    metric_values: Vec<u32>,
}

pub(crate) struct DistrictMetricWriter {
    writer: BatchedWriter<File>,
    district_ids: Vec<u16>,
    batch_rows: usize,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
    district_columns: Vec<Vec<Option<f64>>>,
}

impl F64MetricWriter {
    pub(crate) fn new(
        file: File,
        metric_column_name: impl Into<String>,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> crate::error::Result<Self> {
        let metric_column_name = metric_column_name.into();
        let empty_df = empty_f64_metric_df(&metric_column_name)?;
        let writer = ParquetWriter::new(file)
            .with_compression(compression)
            .batched(empty_df.schema())?;

        Ok(Self {
            writer,
            metric_column_name,
            batch_rows,
            sample_numbers: Vec::with_capacity(batch_rows),
            n_reps_numbers: Vec::with_capacity(batch_rows),
            accepted_numbers: Vec::with_capacity(batch_rows),
            metric_values: Vec::with_capacity(batch_rows),
        })
    }

    pub(crate) fn push(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted_count: u64,
        value: f64,
    ) -> crate::error::Result<()> {
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
        self.flush()?;
        self.writer.finish()?;
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
        self.writer.write_batch(&df)?;
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
        file: File,
        key_column_name: impl Into<String>,
        metric_column_name: impl Into<String>,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> crate::error::Result<Self> {
        let key_column_name = key_column_name.into();
        let metric_column_name = metric_column_name.into();
        let empty_df = empty_u32_keyed_metric_df(&key_column_name, &metric_column_name)?;
        let writer = ParquetWriter::new(file)
            .with_compression(compression)
            .batched(empty_df.schema())?;

        Ok(Self {
            writer,
            key_column_name,
            metric_column_name,
            batch_rows,
            sample_numbers: Vec::with_capacity(batch_rows),
            n_reps_numbers: Vec::with_capacity(batch_rows),
            accepted_numbers: Vec::with_capacity(batch_rows),
            metric_keys: Vec::with_capacity(batch_rows),
            metric_values: Vec::with_capacity(batch_rows),
        })
    }

    pub(crate) fn push(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted_count: u64,
        key: impl Into<String>,
        value: u32,
    ) -> crate::error::Result<()> {
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
        self.flush()?;
        self.writer.finish()?;
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
        self.writer.write_batch(&df)?;
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
        file: File,
        district_ids: Vec<u16>,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> crate::error::Result<Self> {
        let empty_df = empty_district_metric_df(&district_ids)?;
        let writer = ParquetWriter::new(file)
            .with_compression(compression)
            .batched(empty_df.schema())?;

        Ok(Self {
            writer,
            district_columns: district_ids
                .iter()
                .map(|_| Vec::with_capacity(batch_rows))
                .collect(),
            district_ids,
            batch_rows,
            sample_numbers: Vec::with_capacity(batch_rows),
            n_reps_numbers: Vec::with_capacity(batch_rows),
            accepted_numbers: Vec::with_capacity(batch_rows),
        })
    }

    pub(crate) fn push_row_with(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted_count: u64,
        mut value_for_district: impl FnMut(u16) -> Option<f64>,
    ) -> crate::error::Result<()> {
        self.sample_numbers.push(step);
        self.n_reps_numbers.push(n_reps);
        self.accepted_numbers.push(accepted_count);

        for (column_index, &district_id) in self.district_ids.iter().enumerate() {
            self.district_columns[column_index].push(value_for_district(district_id));
        }

        if self.sample_numbers.len() >= self.batch_rows {
            self.flush()?;
        }

        Ok(())
    }

    pub(crate) fn finish(mut self) -> crate::error::Result<()> {
        self.flush()?;
        self.writer.finish()?;
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
        self.writer.write_batch(&df)?;
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
    use tempfile::NamedTempFile;

    #[test]
    fn district_metric_writer_appends_multiple_batches() {
        let file = NamedTempFile::new().unwrap();
        let mut writer = DistrictMetricWriter::new(
            File::create(file.path()).unwrap(),
            vec![1, 2],
            parquet_compression(false),
            2,
        )
        .unwrap();

        writer
            .push_row_with(1, 1, 1, |district| match district {
                1 => Some(0.1),
                2 => Some(0.3),
                _ => unreachable!(),
            })
            .unwrap();
        writer
            .push_row_with(2, 1, 2, |district| match district {
                1 => Some(0.2),
                2 => Some(0.4),
                _ => unreachable!(),
            })
            .unwrap();
        writer
            .push_row_with(3, 2, 3, |district| match district {
                1 => Some(0.5),
                2 => None,
                _ => unreachable!(),
            })
            .unwrap();
        writer.finish().unwrap();

        let df = ParquetReader::new(&mut File::open(file.path()).unwrap())
            .finish()
            .unwrap();
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
}
