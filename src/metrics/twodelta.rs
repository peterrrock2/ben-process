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
    pub(crate) fn refresh(&mut self, changes: &[(usize, u16, u16)]) {
        self.gen += 1;
        for &(node, _old, new) in changes {
            self.stamp[node] = self.gen;
            self.new_label[node] = new;
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
