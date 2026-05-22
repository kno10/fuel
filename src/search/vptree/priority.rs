use std::collections::BinaryHeap;

use super::{SearchCandidate, VPTree};
use crate::api::{NodePoints, SearchFilter};
use crate::{DistPair, DistanceSearch, Float};

struct PointFilter<P> {
    skip_point: P,
}

impl<P> SearchFilter for PointFilter<P>
where
    P: FnMut(usize) -> bool,
{
    fn skip_point(&mut self, index: usize) -> bool { (self.skip_point)(index) }
}

// The priority queue entries used by the VP-tree searcher are different
// from the simple `(dist, point)` pairs used elsewhere.  We keep a local
// definition here rather than reusing `cluster::hierarchical::common::
// QueueEntry` because we need both left‑ and right‑child indices.
#[derive(Debug, Clone, Copy, PartialEq)]
struct QueueEntry<F> {
    distance: F,
    left: usize,
    right: usize,
}

impl<F> QueueEntry<F> {
    const fn new(distance: F, left: usize, right: usize) -> Self { Self { distance, left, right } }
}

impl<F: PartialEq> Eq for QueueEntry<F> {}

impl<F: PartialOrd + PartialEq> PartialOrd for QueueEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl<F: PartialOrd + PartialEq> Ord for QueueEntry<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.distance.partial_cmp(&self.distance).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Priority searcher for incremental nearest neighbor search
//
// The searcher normally returns `DistPair` values with the *exact*
// distance from the query point, in the order they are discovered;
// so not necessarily in ascending order.
// Furthermore, it is possible to filter some results to avoid
// some distance computations.
pub struct PrioritySearcher<'a, F: Float> {
    tree: &'a VPTree<F>,
    heap: BinaryHeap<QueueEntry<F>>,
    threshold: F,
    skip_threshold: F,

    /* state for the node we have popped from the queue but not yet
     * completely handled. `current_node_left`/`right` define the range
     * of the node, and `current_node_dist` is the lower bound on the
     * distance between the query point and any element of that range.
     *
     * We maintain `current_vp_dist` as an `Option` so that the expensive
     * call to `query_distance` can be deferred until the caller actually
     * needs the precise distance.  The flag `children_pushed` ensures we
     * only expand the node once; children are generated using either the
     * bound or the real distance, whichever is available.
     */
    current_node_left: Option<usize>,
    current_node_right: usize,
    has_current_candidate: bool,
    current_node_dist: F,
    current_vp_dist: Option<F>,
}

impl<'a, F: Float> PrioritySearcher<'a, F> {
    pub(super) fn new(tree: &'a VPTree<F>) -> Self {
        let mut searcher = Self {
            tree,
            heap: BinaryHeap::new(),
            threshold: F::infinity(),
            skip_threshold: F::zero(),
            current_node_left: None,
            current_node_right: 0,
            has_current_candidate: false,
            current_node_dist: F::zero(),
            current_vp_dist: None,
        };

        // Initialize with root node
        searcher.heap.push(QueueEntry::new(F::zero(), 0, searcher.tree.points.len()));
        searcher
    }

    /// Reset this searcher for a new query while reusing internal allocations.
    pub fn reset(&mut self) {
        self.threshold = F::infinity();
        self.skip_threshold = F::zero();
        self.current_node_left = None;
        self.current_node_right = 0;
        self.has_current_candidate = false;
        self.current_node_dist = F::zero();
        self.current_vp_dist = None;
        self.heap.clear();
        self.heap.push(QueueEntry::new(F::zero(), 0, self.tree.points.len()));
    }

    /// Reset this searcher for a new query and initialize search bounds.
    pub fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        debug_assert!(skip >= F::zero(), "Skip threshold must be non-negative.");
        debug_assert!(cutoff >= skip, "Cutoff must be >= skip threshold.");
        self.reset();
        self.threshold = cutoff;
        self.skip_threshold = skip;
    }

    /// Pop the next queue entry and prepare it for candidate production.
    /// Children are *not* pushed here; they will be generated when the
    /// candidate is processed so that computing `query_distance` can be
    /// deferred until necessary.
    fn advance_queue(&mut self) -> bool {
        if let Some(entry) = self.heap.pop() {
            if entry.distance > self.threshold {
                self.heap.clear();
                return false;
            }

            self.current_node_dist = entry.distance;
            self.current_node_left = Some(entry.left);
            self.current_node_right = entry.right;
            self.current_vp_dist = None;
            self.has_current_candidate = true;
            return true;
        }

        self.current_node_left = None;
        self.has_current_candidate = false;
        false
    }

    /// Helper to push children of the current node into the heap.  The
    /// bound used for each child is based on the available information; if
    /// `current_vp_dist` has already been computed we can produce the tight
    /// bound, otherwise we fall back to the looser `current_node_dist`.
    fn push_children(&mut self) {
        let left = self.current_node_left.expect("current node must be set");
        let right = self.current_node_right;
        if left + 1 >= right {
            return;
        }
        let mid = usize::midpoint(left, right);

        if left + 1 < mid {
            let left_child = left + 1;
            let child = self.tree.bounds[left_child];
            let (min_dist, max_dist) = if let Some(vp_dist) = self.current_vp_dist {
                let max_dist = vp_dist + child.upper;
                let min_dist =
                    (vp_dist - child.upper).max(child.lower - vp_dist).max(self.current_node_dist);
                (min_dist, max_dist)
            } else {
                (self.current_node_dist, F::infinity())
            };
            if min_dist <= self.threshold && max_dist >= self.skip_threshold {
                self.heap.push(QueueEntry::new(min_dist, left_child, mid));
            }
        }

        if mid < right {
            let right_child = mid;
            let child = self.tree.bounds[right_child];
            let (min_dist, max_dist) = if let Some(vp_dist) = self.current_vp_dist {
                let max_dist = vp_dist + child.upper;
                let min_dist =
                    (vp_dist - child.upper).max(child.lower - vp_dist).max(self.current_node_dist);
                (min_dist, max_dist)
            } else {
                (self.current_node_dist, F::infinity())
            };
            if min_dist <= self.threshold && max_dist >= self.skip_threshold {
                self.heap.push(QueueEntry::new(min_dist, right_child, right));
            }
        }
    }

    /// Decrease search cutoff; values must only decrease.
    pub fn decrease_cutoff(&mut self, threshold: F) {
        debug_assert!(threshold <= self.threshold, "Thresholds must only decrease.");
        self.threshold = threshold;
        if threshold < self.heap.peek().map_or(F::zero(), |entry| entry.distance) {
            self.heap.clear();
        }
    }

    /// Decrease search cutoff if the given value is smaller than the current one.
    ///
    /// Unlike [`decrease_cutoff`](Self::decrease_cutoff), this never panics
    /// when the value is larger than the current cutoff - it simply does nothing.
    /// This is useful for persistent searchers whose cutoff may already be
    /// tighter than the caller's local threshold.
    pub fn try_decrease_cutoff(&mut self, threshold: F) {
        if threshold < self.threshold {
            self.threshold = threshold;
            if threshold < self.heap.peek().map_or(F::zero(), |entry| entry.distance) {
                self.heap.clear();
            }
        }
    }

    /// Increase lower skip threshold; values must only increase.
    pub fn increase_skip(&mut self, threshold: F) {
        debug_assert!(threshold >= self.skip_threshold, "Skip thresholds must only increase.");
        self.skip_threshold = threshold;
    }

    /// Lower bound of all remaining candidates.
    pub fn all_lower_bound(&self) -> F {
        if self.has_current_candidate {
            self.current_node_dist
        } else {
            self.heap.peek().map_or(F::infinity(), |entry| entry.distance)
        }
    }
}

impl<F: Float> PrioritySearcher<'_, F> {
    /// Like `next_candidate`, but consults a filter before evaluating a node.
    ///
    /// `skip_node` can prune an entire subtree before any exact distance is
    /// computed. `skip_point` can reject just the pivot while still exploring
    /// the node's children.
    /// Like `next_candidate`, but consults a filter before evaluating a node.
    ///
    /// # Panics
    ///
    /// - if the internal state is corrupted and `current_node_left` is not set when
    ///   `has_current_candidate` is true.
    pub fn next_with_filter<D: DistanceSearch<F> + ?Sized, S>(
        &mut self, query: &D, filter: &mut S,
    ) -> Option<DistPair<F>>
    where
        S: SearchFilter + ?Sized,
    {
        loop {
            if self.has_current_candidate {
                let node_idx = self.current_node_left.expect("current node must be set");
                let right = self.current_node_right;
                if filter.skip_node(NodePoints::new(&self.tree.points[node_idx..right])) {
                    self.has_current_candidate = false;
                    continue;
                }

                let vp = self.tree.points[node_idx] as usize;

                // decide whether the caller is even interested in this pivot
                let skip = filter.skip_point(vp);

                if skip {
                    // We intentionally keep the lazy path for skipped pivots so
                    // filters can reject them without forcing an exact distance
                    // computation. Child bounds fall back to the node bound.
                    self.push_children();

                    // discard this candidate
                    self.has_current_candidate = false;
                    continue;
                }

                // For returned candidates we will need the exact pivot distance
                // anyway, so compute it before pushing children. This yields the
                // tight subtree bounds required for cutoff/skip pruning and for
                // `all_lower_bound` to track the remaining queue accurately.
                let vp_dist = *self.current_vp_dist.get_or_insert_with(|| query.query_distance(vp));

                self.push_children();

                self.has_current_candidate = false;

                if vp_dist <= self.threshold
                    && (self.skip_threshold == F::zero() || vp_dist >= self.skip_threshold)
                {
                    return Some(DistPair::new(vp_dist, vp));
                }
                continue;
            }

            if !self.advance_queue() {
                return None;
            }
        }
    }

    /// Like `next_with_filter`, but also returns the lower bound associated with
    /// the candidate.
    pub fn next_with_filter_bounds<D: DistanceSearch<F> + ?Sized, S>(
        &mut self, query: &D, filter: &mut S,
    ) -> Option<SearchCandidate<F>>
    where
        S: SearchFilter,
    {
        loop {
            if self.has_current_candidate {
                let node_idx = self.current_node_left.expect("current node must be set");
                let right = self.current_node_right;
                if filter.skip_node(NodePoints::new(&self.tree.points[node_idx..right])) {
                    self.has_current_candidate = false;
                    continue;
                }

                let vp = self.tree.points[node_idx] as usize;

                let skip = filter.skip_point(vp);

                if skip {
                    self.push_children();
                    self.has_current_candidate = false;
                    continue;
                }

                let lower_bound = self.current_node_dist;
                let vp_dist = *self.current_vp_dist.get_or_insert_with(|| query.query_distance(vp));

                self.push_children();
                self.has_current_candidate = false;

                if vp_dist <= self.threshold
                    && (self.skip_threshold == F::zero() || vp_dist >= self.skip_threshold)
                {
                    return Some(SearchCandidate::new(vp_dist, lower_bound, vp));
                }
                continue;
            }

            if !self.advance_queue() {
                return None;
            }
        }
    }

    /// Like `next_candidate`, but skips pivot points selected by the supplied
    /// predicate without evaluating their exact query distance.
    /// Basic candidate generator with no filtering.
    pub fn next<D: DistanceSearch<F> + ?Sized>(&mut self, query: &D) -> Option<DistPair<F>> {
        self.next_filtered(query, |_| false)
    }

    /// Like `next_filtered`, but skips pivot points selected by the supplied
    /// predicate without evaluating their exact query distance.
    pub fn next_filtered<D: DistanceSearch<F> + ?Sized, P>(
        &mut self, query: &D, skip_pred: P,
    ) -> Option<DistPair<F>>
    where
        P: FnMut(usize) -> bool,
    {
        self.next_with_filter(query, &mut PointFilter { skip_point: skip_pred })
    }

    /// Like `next_filtered`, but also returns the lower bound associated with
    /// the candidate.
    pub fn next_filtered_bounds<D: DistanceSearch<F> + ?Sized, P>(
        &mut self, query: &D, skip_pred: P,
    ) -> Option<SearchCandidate<F>>
    where
        P: FnMut(usize) -> bool,
    {
        self.next_with_filter_bounds(query, &mut PointFilter { skip_point: skip_pred })
    }
}
