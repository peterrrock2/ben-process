use std::io;

/// Upper bound on district ids supported by the dense bitmask path.
///
/// `u128` stores one observed bit per district id, so district labels must stay below 128 until
/// this representation is widened.
pub(crate) const MAX_DISTRICTS: u16 = 128;

/// Fold a single district id into an observed bitmask, panicking past the dense-bitmask limit.
///
/// Shared by every metric so the set capture is bound-checked identically wherever it happens — a
/// raw `1u128 << district_id` would silently wrap (`1 << 128 == 1`) in release builds for ids >=
/// 128.
#[inline]
pub(crate) fn observe_district(observed: &mut u128, district_id: u16) {
    if district_id >= MAX_DISTRICTS {
        panic!(
            "district id {} exceeds current {}-district limit; widen the observed bitmask",
            district_id, MAX_DISTRICTS
        );
    }
    *observed |= 1u128 << district_id;
}

/// Returns `(n_districts, observed_mask)` for an assignment.
///
/// `n_districts` is `max(assignment) + 1`, preserving the existing dense-buffer shape used by
/// metric hot loops. `observed_mask` has bit `d` set when district `d` appears in the assignment.
pub(crate) fn observed_assignment_districts(assignment: &[u16]) -> (u16, u128) {
    let mut observed: u128 = 0;
    let mut max_d: u16 = 0;
    for &district_id in assignment {
        observe_district(&mut observed, district_id);
        if district_id > max_d {
            max_d = district_id;
        }
    }

    (max_d + 1, observed)
}

/// Bits set in `mask`, returned in ascending order.
pub(crate) fn sorted_district_ids(mut mask: u128) -> Vec<u16> {
    let mut out = Vec::with_capacity(mask.count_ones() as usize);
    while mask != 0 {
        out.push(mask.trailing_zeros() as u16);
        mask &= mask - 1;
    }
    out
}

#[cfg(test)]
pub(crate) fn assert_district_set_unchanged(observed: u128, expected: u128, output_name: &str) {
    validate_district_set_unchanged(observed, expected, output_name)
        .expect("a changed district set should error in the assertion wrapper");
}

/// Enforce that a plan's district label set exactly matches the first assignment's.
///
/// The streaming per-district Parquet schema is fixed from the first plan, so a later plan that
/// *adds* a district id would need a new column and one that *drops* a district id would leave a
/// column with no value. A valid ensemble keeps the same district labels in every plan, so either
/// direction is a hard error — we refuse to paper over it with extra columns, nulls, or zeros.
pub(crate) fn validate_district_set_unchanged(
    observed: u128,
    expected: u128,
    output_name: &str,
) -> io::Result<()> {
    if observed == expected {
        return Ok(());
    }

    let unseen = observed & !expected;
    let missing = expected & !observed;
    let mut parts: Vec<String> = Vec::new();
    if unseen != 0 {
        parts.push(format!(
            "encountered districts {:?} not present in first assignment",
            sorted_district_ids(unseen)
        ));
    }
    if missing != 0 {
        parts.push(format!(
            "districts {:?} from the first assignment are missing from a later plan",
            sorted_district_ids(missing)
        ));
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "{}; every plan in the ensemble must use the same district labels to stream {} output with a fixed schema",
            parts.join("; "),
            output_name
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        assert_district_set_unchanged, observed_assignment_districts, sorted_district_ids,
    };

    #[test]
    fn observed_assignment_districts_returns_dense_count_and_mask() {
        let (n_districts, observed) = observed_assignment_districts(&[1, 3, 1]);

        assert_eq!(n_districts, 4);
        assert_eq!(observed, (1u128 << 1) | (1u128 << 3));
    }

    #[test]
    #[should_panic(expected = "district id 128 exceeds current 128-district limit")]
    fn observed_assignment_districts_panics_when_assignment_exceeds_supported_limit() {
        let _ = observed_assignment_districts(&[128]);
    }

    #[test]
    fn sorted_district_ids_returns_ascending_order() {
        let mask = (1u128 << 63) | (1u128 << 1) | (1u128 << 65);
        assert_eq!(sorted_district_ids(mask), vec![1, 63, 65]);
    }

    #[test]
    #[should_panic(
        expected = "encountered districts [3] not present in first assignment; every plan in the ensemble must use the same district labels to stream tally output with a fixed schema"
    )]
    fn assert_district_set_unchanged_rejects_added_district() {
        assert_district_set_unchanged((1u128 << 1) | (1u128 << 3), 1u128 << 1, "tally");
    }

    #[test]
    #[should_panic(
        expected = "districts [3] from the first assignment are missing from a later plan; every plan in the ensemble must use the same district labels to stream tally output with a fixed schema"
    )]
    fn assert_district_set_unchanged_rejects_dropped_district() {
        // Expected set {1,3}; this plan only has district 1, so district 3 has vanished. A valid
        // ensemble keeps the same labels in every plan, so this must error rather than emit a
        // null/zero column for district 3.
        assert_district_set_unchanged(1u128 << 1, (1u128 << 1) | (1u128 << 3), "tally");
    }

    #[test]
    fn assert_district_set_unchanged_accepts_identical_sets() {
        // Exact match is the only accepted case.
        assert_district_set_unchanged(
            (1u128 << 1) | (1u128 << 3),
            (1u128 << 1) | (1u128 << 3),
            "tally",
        );
    }
}
