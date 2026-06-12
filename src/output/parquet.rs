//! Batched Parquet writers for the per-plan metric modes.
//!
//! One writer, [`BatchedMetricWriter`], owns the whole output lifecycle — lazy file creation,
//! the shared `step` / `n_reps` / `accepted_count` prefix columns, batch flushing, and the
//! zero-row fallback. What varies per mode is only the metric columns, supplied as a
//! [`MetricColumns`] implementation; the three column shapes in use are aliased as
//! [`F64MetricWriter`], [`U32KeyedMetricWriter`], and [`DistrictMetricWriter`].
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
use std::mem;

/// Deferred output-file constructor. Invoked at most once, on the first pushed row (or at
/// `finish` for a zero-row run). The factory owns any directory creation its path needs.
pub(crate) type FileFactory = Box<dyn FnOnce() -> io::Result<File>>;

/// The metric-column part of an output table: everything to the right of the shared
/// `step` / `n_reps` / `accepted_count` prefix.
///
/// Implementations buffer one cell set per [`push`](Self::push) and surrender the buffered
/// columns at each batch flush. The buffers must stay row-aligned with the writer's prefix
/// buffers — exactly one `push` per writer row.
pub(crate) trait MetricColumns {
    /// Per-row payload accepted by [`BatchedMetricWriter::push_row`].
    type Row<'a>;

    /// Fix schema-affecting state from the first row. Called exactly once, right before the
    /// output file and its schema are created; shapes with a static schema use the default no-op.
    fn fix_schema(&mut self, _first_row: &Self::Row<'_>, _batch_rows: usize) {}

    /// The metric columns of the empty-schema dataframe (zero rows each).
    fn empty_columns(&self) -> Vec<Column>;

    /// Buffer one row's metric cells.
    fn push(&mut self, row: Self::Row<'_>);

    /// Drain the buffered cells into columns for one batch, leaving fresh buffers with
    /// `batch_rows` capacity behind.
    fn take_columns(&mut self, batch_rows: usize) -> Vec<Column>;
}

/// Streaming Parquet writer for per-plan metric tables: the shared prefix columns plus whatever
/// metric columns `C` contributes. See the module docs for the lazy file-creation contract.
pub(crate) struct BatchedMetricWriter<C: MetricColumns> {
    columns: C,
    make_file: Option<FileFactory>,
    writer: Option<BatchedWriter<File>>,
    compression: ParquetCompression,
    batch_rows: usize,
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
}

impl<C: MetricColumns> BatchedMetricWriter<C> {
    fn with_columns(
        columns: C,
        make_file: FileFactory,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> Self {
        Self {
            columns,
            make_file: Some(make_file),
            writer: None,
            compression,
            batch_rows,
            sample_numbers: Vec::with_capacity(batch_rows),
            n_reps_numbers: Vec::with_capacity(batch_rows),
            accepted_numbers: Vec::with_capacity(batch_rows),
        }
    }

    fn create_writer(&mut self) -> crate::error::Result<()> {
        let make_file = self
            .make_file
            .take()
            .expect("output file factory should be unconsumed before the writer exists");
        let file = make_file()?;
        let mut schema_columns = prefix_columns(Vec::new(), Vec::new(), Vec::new());
        schema_columns.extend(self.columns.empty_columns());
        let empty_df = DataFrame::new_infer_height(schema_columns)?;
        self.writer = Some(
            ParquetWriter::new(file)
                .with_compression(self.compression)
                .batched(empty_df.schema())?,
        );
        Ok(())
    }

    /// Push one plan's row. The first call fixes the schema (for shapes that derive it from the
    /// first row) and creates the output file.
    pub(crate) fn push_row(
        &mut self,
        step: u64,
        n_reps: u32,
        accepted_count: u64,
        row: C::Row<'_>,
    ) -> crate::error::Result<()> {
        if self.writer.is_none() {
            self.columns.fix_schema(&row, self.batch_rows);
            self.create_writer()?;
        }

        self.sample_numbers.push(step);
        self.n_reps_numbers.push(n_reps);
        self.accepted_numbers.push(accepted_count);
        self.columns.push(row);

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

        let cap = self.batch_rows;
        let mut batch_columns = prefix_columns(
            mem::replace(&mut self.sample_numbers, Vec::with_capacity(cap)),
            mem::replace(&mut self.n_reps_numbers, Vec::with_capacity(cap)),
            mem::replace(&mut self.accepted_numbers, Vec::with_capacity(cap)),
        );
        batch_columns.extend(self.columns.take_columns(cap));
        let df = DataFrame::new_infer_height(batch_columns)?;
        self.writer
            .as_mut()
            .expect("writer should exist once rows are buffered")
            .write_batch(&df)?;

        Ok(())
    }
}

fn prefix_columns(
    sample_numbers: Vec<u64>,
    n_reps_numbers: Vec<u32>,
    accepted_numbers: Vec<u64>,
) -> Vec<Column> {
    vec![
        Series::new("step".into(), sample_numbers).into(),
        Series::new("n_reps".into(), n_reps_numbers).into(),
        Series::new("accepted_count".into(), accepted_numbers).into(),
    ]
}

/// One named `f64` column; rows are the bare value.
pub(crate) struct F64Column {
    name: String,
    values: Vec<f64>,
}

impl MetricColumns for F64Column {
    type Row<'a> = f64;

    fn empty_columns(&self) -> Vec<Column> {
        vec![Series::new(self.name.as_str().into(), Vec::<f64>::new()).into()]
    }

    fn push(&mut self, value: f64) {
        self.values.push(value);
    }

    fn take_columns(&mut self, batch_rows: usize) -> Vec<Column> {
        vec![Series::new(
            self.name.as_str().into(),
            mem::replace(&mut self.values, Vec::with_capacity(batch_rows)),
        )
        .into()]
    }
}

pub(crate) type F64MetricWriter = BatchedMetricWriter<F64Column>;

impl F64MetricWriter {
    pub(crate) fn new(
        make_file: FileFactory,
        metric_column_name: impl Into<String>,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> Self {
        Self::with_columns(
            F64Column {
                name: metric_column_name.into(),
                values: Vec::with_capacity(batch_rows),
            },
            make_file,
            compression,
            batch_rows,
        )
    }
}

/// A string key column plus a named `u32` column; rows are `(key, value)`.
pub(crate) struct U32KeyedColumns {
    key_name: String,
    metric_name: String,
    keys: Vec<String>,
    values: Vec<u32>,
}

impl MetricColumns for U32KeyedColumns {
    type Row<'a> = (String, u32);

    fn empty_columns(&self) -> Vec<Column> {
        vec![
            Series::new(self.key_name.as_str().into(), Vec::<String>::new()).into(),
            Series::new(self.metric_name.as_str().into(), Vec::<u32>::new()).into(),
        ]
    }

    fn push(&mut self, (key, value): (String, u32)) {
        self.keys.push(key);
        self.values.push(value);
    }

    fn take_columns(&mut self, batch_rows: usize) -> Vec<Column> {
        vec![
            Series::new(
                self.key_name.as_str().into(),
                mem::replace(&mut self.keys, Vec::with_capacity(batch_rows)),
            )
            .into(),
            Series::new(
                self.metric_name.as_str().into(),
                mem::replace(&mut self.values, Vec::with_capacity(batch_rows)),
            )
            .into(),
        ]
    }
}

pub(crate) type U32KeyedMetricWriter = BatchedMetricWriter<U32KeyedColumns>;

impl U32KeyedMetricWriter {
    pub(crate) fn new(
        make_file: FileFactory,
        key_column_name: impl Into<String>,
        metric_column_name: impl Into<String>,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> Self {
        Self::with_columns(
            U32KeyedColumns {
                key_name: key_column_name.into(),
                metric_name: metric_column_name.into(),
                keys: Vec::with_capacity(batch_rows),
                values: Vec::with_capacity(batch_rows),
            },
            make_file,
            compression,
            batch_rows,
        )
    }
}

/// One nullable `district_N` column per observed district; rows are `(observed, values)` where
/// `values` is indexed by district label (`values[d]` is district `d`'s value, length
/// `max observed label + 1`) — the shape selects the observed labels itself.
///
/// The district-column schema is fixed from the **first pushed row's** observed set; callers
/// never pass district ids. This is sound because `run_pipeline` enforces a fixed district set
/// across the ensemble, so the first row's set is every row's set.
pub(crate) struct DistrictColumns {
    district_ids: Vec<u16>,
    columns: Vec<Vec<Option<f64>>>,
}

impl MetricColumns for DistrictColumns {
    type Row<'a> = (u128, &'a [f64]);

    fn fix_schema(&mut self, &(observed, _values): &Self::Row<'_>, batch_rows: usize) {
        self.district_ids = sorted_district_ids(observed);
        self.columns = self
            .district_ids
            .iter()
            .map(|_| Vec::with_capacity(batch_rows))
            .collect();
    }

    fn empty_columns(&self) -> Vec<Column> {
        self.district_ids
            .iter()
            .map(|district_id| {
                Series::new(
                    format!("district_{}", district_id).into(),
                    Vec::<Option<f64>>::new(),
                )
                .into()
            })
            .collect()
    }

    fn push(&mut self, (_observed, values): (u128, &[f64])) {
        for (column_index, &district_id) in self.district_ids.iter().enumerate() {
            self.columns[column_index].push(values.get(district_id as usize).copied());
        }
    }

    fn take_columns(&mut self, batch_rows: usize) -> Vec<Column> {
        self.district_ids
            .iter()
            .zip(self.columns.iter_mut())
            .map(|(district_id, column)| {
                Series::new(
                    format!("district_{}", district_id).into(),
                    mem::replace(column, Vec::with_capacity(batch_rows)),
                )
                .into()
            })
            .collect()
    }
}

pub(crate) type DistrictMetricWriter = BatchedMetricWriter<DistrictColumns>;

impl DistrictMetricWriter {
    pub(crate) fn new(
        make_file: FileFactory,
        compression: ParquetCompression,
        batch_rows: usize,
    ) -> Self {
        Self::with_columns(
            DistrictColumns {
                district_ids: Vec::new(),
                columns: Vec::new(),
            },
            make_file,
            compression,
            batch_rows,
        )
    }
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
            .push_row(1, 1, 1, (observed, &[9.9, 0.1, 0.3]))
            .unwrap();
        assert!(path.exists(), "first pushed row creates the output file");
        writer
            .push_row(2, 1, 2, (observed, &[9.9, 0.2, 0.4]))
            .unwrap();
        writer.push_row(3, 2, 3, (observed, &[9.9, 0.5])).unwrap();
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
