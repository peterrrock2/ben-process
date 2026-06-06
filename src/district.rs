/// Upper bound on district ids supported by the dense bitmask path.
///
/// `u128` stores one observed bit per district id, so district labels must stay below 128 until
/// this representation is widened.
pub(crate) const MAX_DISTRICTS: u16 = 128;

/// Returns `(n_districts, observed_mask)` for an assignment.
///
/// `n_districts` is `max(assignment) + 1`, preserving the existing dense-buffer shape used by
/// metric hot loops. `observed_mask` has bit `d` set when district `d` appears in the assignment.
pub(crate) fn observed_assignment_districts(assignment: &[u16]) -> (u16, u128) {
    let mut observed: u128 = 0;
    let mut max_d: u16 = 0;
    for &district_id in assignment {
        if district_id >= MAX_DISTRICTS {
            panic!(
                "district id {} exceeds current {}-district limit; widen the observed bitmask",
                district_id, MAX_DISTRICTS
            );
        }
        observed |= 1u128 << district_id;
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

pub(crate) fn assert_no_unseen_districts(observed: u128, expected: u128, output_name: &str) {
    let unseen = observed & !expected;
    if unseen != 0 {
        panic!(
            "encountered districts {:?} not present in first assignment; cannot stream {} output with a fixed schema",
            sorted_district_ids(unseen),
            output_name
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{assert_no_unseen_districts, observed_assignment_districts, sorted_district_ids};

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
        expected = "encountered districts [3] not present in first assignment; cannot stream tally output with a fixed schema"
    )]
    fn assert_no_unseen_districts_panics_with_context() {
        assert_no_unseen_districts((1u128 << 1) | (1u128 << 3), 1u128 << 1, "tally");
    }
}
