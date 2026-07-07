use crate::district::validate_district_set_unchanged;
use crate::error::{Error, Result};
use crate::input::BenSource;
use crate::pipeline::{capped_reps, make_progress_bar};
use ben::io::reader::TwoDeltaFrameEvent;
use std::io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaChange {
    pub(crate) node: usize,
    pub(crate) old: u16,
    pub(crate) new: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TwoDeltaRow {
    pub(crate) step: u64,
    pub(crate) n_reps: u32,
    pub(crate) accepted: u64,
}

pub(crate) trait IncrementalTwoDeltaMetric {
    fn seed(&mut self, assignment: &[u16]) -> Result<()>;
    fn update_delta(&mut self, before: &[u16], changes: &[DeltaChange]) -> Result<()>;
    fn observed(&self) -> u128;
}

pub(crate) struct TwoDeltaRunOptions<'a> {
    pub(crate) expected_len: usize,
    pub(crate) expected_len_label: &'static str,
    pub(crate) output_name: &'a str,
    pub(crate) show_progress: bool,
    pub(crate) max_samples: Option<usize>,
}

fn validate_changes(before: &[u16], changes: &[(u32, u16, u16)]) -> Result<Vec<DeltaChange>> {
    let mut validated = Vec::with_capacity(changes.len());
    for &(node, old, new) in changes {
        let node = node as usize;
        let Some(&current) = before.get(node) else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "TwoDelta delta references node {node} outside assignment length {}",
                    before.len()
                ),
            )
            .into());
        };
        if current != old {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "TwoDelta delta old label mismatch at node {node}: expected {current}, got {old}",
                ),
            )
            .into());
        }
        validated.push(DeltaChange { node, old, new });
    }
    Ok(validated)
}

pub(crate) fn run_incremental_twodelta<M, W>(
    source: &BenSource,
    options: TwoDeltaRunOptions<'_>,
    metric: &mut M,
    mut write: W,
) -> Result<()>
where
    M: IncrementalTwoDeltaMetric,
    W: FnMut(&mut M, TwoDeltaRow) -> Result<()>,
{
    let progress_bar = if options.show_progress {
        Some(make_progress_bar(match options.max_samples {
            Some(n) => n,
            None => source.count_samples()?,
        }))
    } else {
        None
    };

    let mut remaining_samples = options.max_samples;
    let mut assignment: Option<Vec<u16>> = None;
    let mut expected_observed: Option<u128> = None;
    let mut step = 1u64;

    for (accepted, event) in (1u64..).zip(source.open_reader()?.into_twodelta_events()) {
        if remaining_samples == Some(0) {
            break;
        }

        let n_reps = match event? {
            TwoDeltaFrameEvent::Snapshot {
                assignment: snapshot,
                count,
                ..
            } => {
                if snapshot.len() != options.expected_len {
                    return Err(Error::AssignmentLength {
                        actual: snapshot.len(),
                        actual_label: "BEN assignment length",
                        expected: options.expected_len,
                        expected_label: options.expected_len_label,
                    });
                }
                metric.seed(&snapshot)?;
                assignment = Some(snapshot);
                capped_reps(&mut remaining_samples, count)
            }
            TwoDeltaFrameEvent::Delta { changes, count } => {
                let assignment = assignment.as_mut().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "TwoDelta delta event appeared before an initial snapshot",
                    )
                })?;
                let changes = validate_changes(assignment, &changes)?;
                metric.update_delta(assignment, &changes)?;
                for change in changes {
                    assignment[change.node] = change.new;
                }
                capped_reps(&mut remaining_samples, count)
            }
        };

        match expected_observed {
            None => expected_observed = Some(metric.observed()),
            Some(expected) => {
                validate_district_set_unchanged(metric.observed(), expected, options.output_name)?;
            }
        }

        write(
            metric,
            TwoDeltaRow {
                step,
                n_reps: n_reps as u32,
                accepted,
            },
        )?;
        step += n_reps as u64;
        if let Some(progress_bar) = &progress_bar {
            progress_bar.inc(n_reps as u64);
        }
    }

    if let Some(progress_bar) = progress_bar {
        progress_bar.finish_and_clear();
    }
    Ok(())
}

/// Sparse lookup for labels after the current TwoDelta event is applied.
///
/// `stamp` avoids clearing `new_label` for every event: only nodes touched in the current
/// generation override the pre-delta assignment.
pub(crate) struct PostDeltaLabels {
    new_label: Vec<u16>,
    stamp: Vec<u64>,
    gen: u64,
}

impl PostDeltaLabels {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            new_label: vec![0; node_count],
            stamp: vec![0; node_count],
            gen: 0,
        }
    }

    /// Load the changed labels for one delta without clearing the previous scratch arrays.
    pub(crate) fn refresh(&mut self, changes: &[DeltaChange]) {
        self.gen += 1;
        for change in changes {
            self.stamp[change.node] = self.gen;
            self.new_label[change.node] = change.new;
        }
    }

    /// Return the node's post-delta label, falling back to the pre-delta assignment.
    pub(crate) fn label(&self, before: &[u16], node: usize) -> u16 {
        if self.stamp[node] == self.gen {
            self.new_label[node]
        } else {
            before[node]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate_changes;

    #[test]
    fn validate_changes_rejects_old_label_mismatch() {
        let err = validate_changes(&[1, 1, 2], &[(1, 2, 1)]).unwrap_err();

        assert!(
            err.to_string().contains("old label mismatch"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn validate_changes_rejects_node_outside_assignment() {
        let err = validate_changes(&[1, 1, 2], &[(3, 2, 1)]).unwrap_err();

        assert!(
            err.to_string().contains("outside assignment length 3"),
            "unexpected error: {err}",
        );
    }
}
