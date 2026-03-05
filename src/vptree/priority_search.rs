use std::collections::BinaryHeap;

use num_traits::Float;

use crate::DataAccess;

use super::{DistPair, VPTree};

#[derive(Debug, Clone, Copy, PartialEq)]
struct QueueEntry<F = f64> {
    distance: F,
    left: usize,
    right: usize,
}

impl<F> QueueEntry<F> {
    const fn new(distance: F, left: usize, right: usize) -> Self {
        Self {
            distance,
            left,
            right,
        }
    }
}

impl<F: PartialEq> Eq for QueueEntry<F> {}

impl<F: PartialOrd + PartialEq> PartialOrd for QueueEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: PartialOrd + PartialEq> Ord for QueueEntry<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Priority searcher for incremental nearest neighbor search
pub struct PrioritySearcher<'a, D: DataAccess, F = f64> {
    tree: &'a VPTree<F>,
    data: D,
    heap: BinaryHeap<QueueEntry<F>>,
    threshold: F,
    skip_threshold: F,
    current_node_left: Option<usize>,
    has_current_candidate: bool,
    current_node_dist: F,
    current_vp_dist: F,
}

impl<'a, D: DataAccess, F: Float> PrioritySearcher<'a, D, F> {
    pub(super) fn new(tree: &'a VPTree<F>, data: D) -> Self {
        let mut searcher = Self {
            tree,
            data,
            heap: BinaryHeap::new(),
            threshold: F::infinity(),
            skip_threshold: F::zero(),
            current_node_left: None,
            has_current_candidate: false,
            current_node_dist: F::zero(),
            current_vp_dist: F::nan(),
        };

        // Initialize with root node
        searcher
            .heap
            .push(QueueEntry::new(F::zero(), 0, searcher.tree.points.len()));
        searcher
    }

    /// Reset this searcher for a new query while reusing internal allocations.
    pub fn reset(&mut self) {
        self.threshold = F::infinity();
        self.skip_threshold = F::zero();
        self.current_node_left = None;
        self.has_current_candidate = false;
        self.current_node_dist = F::zero();
        self.current_vp_dist = F::nan();
        self.heap.clear();
        self.heap
            .push(QueueEntry::new(F::zero(), 0, self.tree.points.len()));
    }

    /// Reset this searcher for a new query and initialize search bounds.
    ///
    /// # Panics
    ///
    /// Panics if `skip < 0.0` or if `cutoff < skip`.
    pub fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        assert!(skip >= F::zero(), "Skip threshold must be non-negative.");
        assert!(cutoff >= skip, "Cutoff must be >= skip threshold.");
        self.reset();
        self.threshold = cutoff;
        self.skip_threshold = skip;
    }

    /// Replace the data access object and reset search state.
    pub fn reset_with_data(&mut self, data: D) {
        self.data = data;
        self.reset();
    }

    /// Expand the next node from the priority queue.
    fn advance_queue(&mut self) -> bool {
        if let Some(entry) = self.heap.pop() {
            if entry.distance > self.threshold {
                self.heap.clear();
                return false;
            }

            self.current_node_dist = entry.distance;
            self.current_node_left = Some(entry.left);

            let vp = self.tree.points[entry.left];
            self.current_vp_dist = F::from(self.data.query_distance(vp as usize))
                .expect("distance cannot be represented by target float type");

            if entry.left + 1 >= entry.right {
                self.has_current_candidate = true;
                return true;
            }

            let vp_dist = self.current_vp_dist;
            let mid = usize::midpoint(entry.left, entry.right);

            if entry.left + 1 < mid {
                let left_child = entry.left + 1;
                let child = self.tree.bounds[left_child];
                let min_dist = (vp_dist - child.upper)
                    .max(child.lower - vp_dist)
                    .max(entry.distance);

                if min_dist <= self.threshold
                    && (self.skip_threshold == F::zero()
                        || vp_dist + child.upper >= self.skip_threshold)
                {
                    self.heap.push(QueueEntry::new(min_dist, left_child, mid));
                }
            }

            if mid < entry.right {
                let right_child = mid;
                let child = self.tree.bounds[right_child];
                let min_dist = (vp_dist - child.upper)
                    .max(child.lower - vp_dist)
                    .max(entry.distance);

                if min_dist <= self.threshold
                    && (self.skip_threshold == F::zero()
                        || vp_dist + child.upper >= self.skip_threshold)
                {
                    self.heap
                        .push(QueueEntry::new(min_dist, right_child, entry.right));
                }
            }

            self.has_current_candidate = true;
            return true;
        }

        self.current_node_left = None;
        self.has_current_candidate = false;
        false
    }

    /// Retrieve the next candidate in approximately increasing distance order.
    fn next_candidate(&mut self) -> Option<DistPair<F>> {
        loop {
            if self.has_current_candidate {
                self.has_current_candidate = false;
                if self.current_vp_dist <= self.threshold
                    && (self.skip_threshold == F::zero()
                        || self.current_vp_dist >= self.skip_threshold)
                {
                    let node_idx = self.current_node_left.expect("current node must be set");
                    let vp = self.tree.points[node_idx];
                    return Some(DistPair::new(self.current_vp_dist, vp as usize));
                }
            }

            if !self.advance_queue() {
                return None;
            }
        }
    }

    /// Decrease search cutoff; values must only decrease.
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is greater than the current cutoff.
    pub fn decrease_cutoff(&mut self, threshold: F) {
        assert!(threshold <= self.threshold, "Thresholds must only decrease.");
        self.threshold = threshold;
        if threshold < self.current_node_dist {
            self.heap.clear();
        }
    }

    /// Increase lower skip threshold; values must only increase.
    ///
    /// # Panics
    ///
    /// Panics if `threshold` is less than the current skip threshold.
    pub fn increase_skip(&mut self, threshold: F) {
        assert!(
            threshold >= self.skip_threshold,
            "Skip thresholds must only increase."
        );
        self.skip_threshold = threshold;
    }

    /// Backward-compatible alias for setting a search cutoff.
    pub fn set_threshold(&mut self, threshold: F) {
        self.decrease_cutoff(threshold);
    }

    /// Lower bound of all remaining candidates in the queue.
    pub const fn all_lower_bound(&self) -> F {
        self.current_node_dist
    }

    /// Get all neighbors within the current threshold
    pub fn get_all_neighbors(&mut self) -> Vec<DistPair<F>> {
        let mut result = Vec::new();
        for neighbor in self.by_ref() {
            result.push(neighbor);
        }
        result
    }
}

impl<D: DataAccess, T: Float> Iterator for PrioritySearcher<'_, D, T> {
    type Item = DistPair<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_candidate()
    }
}
