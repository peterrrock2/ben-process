pub mod canonical;
pub mod changed_assignments;
pub mod cut_edges;
pub mod extract_unique_plans;
pub mod polsby_popper;
pub mod region;
pub mod reock;
pub mod tally_keys;
pub(crate) mod twodelta;
pub mod unique_plans;

/// Dense per-district tables produced by one prepared metric.
#[derive(Debug, PartialEq)]
pub struct PreparedMetricOutput {
    pub values: Vec<f64>,
    pub table_count: usize,
    pub district_slots: usize,
    pub observed: u128,
}

impl PreparedMetricOutput {
    pub fn table(&self, index: usize) -> Option<&[f64]> {
        let start = index.checked_mul(self.district_slots)?;
        let end = start.checked_add(self.district_slots)?;
        self.values.get(start..end)
    }
}

pub(crate) fn validate_assignment_length(
    assignment: &[u16],
    expected: usize,
    expected_label: &'static str,
    actual_label: &'static str,
) -> crate::error::Result<()> {
    if assignment.len() != expected {
        return Err(crate::error::Error::AssignmentLength {
            actual: assignment.len(),
            actual_label,
            expected,
            expected_label,
        });
    }
    Ok(())
}
