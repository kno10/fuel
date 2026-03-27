//! Incremental best-first cover tree searcher for distance-sorted iteration.
//!
//! Key behaviors:
//! - Node queue stores subtree lower bounds, not actual distances.
//! - Candidate queue stores exact point distances discovered so far.
//! - `all_lower_bound()` provides monotonic lower bound across nodes+candidates.
//! - `decrease_cutoff()` prunes both queues consistently.
//!
//! `skip_threshold` allows efficient selective filtering while still preserving ordered expansion.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::covertree::CoverTree;
use crate::covertree::construct::CoverTreeNode;
use crate::{CandidateHeap, DistPair, DistanceSearch, Float, PrioritySearcher};

#[derive(Debug, Clone, Copy)]
struct NodeEntry<'a, F>
where
    F: Float,
{
    lower_bound: F,
    node: &'a CoverTreeNode<F>,
    emit_center: bool,
}

impl<'a, F: Float> PartialEq for NodeEntry<'a, F> {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound
            .partial_cmp(&other.lower_bound)
            .map(|o| o == Ordering::Equal)
            .unwrap_or(false)
    }
}

impl<'a, F: Float> Eq for NodeEntry<'a, F> {}

impl<'a, F: Float> PartialOrd for NodeEntry<'a, F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // We want smallest lower bound first.
        Some(self.cmp(other))
    }
}

impl<'a, F: Float> Ord for NodeEntry<'a, F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower_bound.partial_cmp(&self.lower_bound).unwrap_or(Ordering::Equal)
    }
}

/// Priority searcher for CoverTree using best-first traversal.
///
/// Traversal is incremental: children are expanded lazily based on lower
/// bounds, and points are emitted in increasing distance order without
/// collecting all nodes at once.
pub struct CoverTreePrioritySearcher<'a, F>
where
    F: Float,
{
    tree: &'a CoverTree<F>,
    node_queue: BinaryHeap<NodeEntry<'a, F>>,
    candidate_queue: CandidateHeap<F>,
    threshold: F,
    skip_threshold: F,
}

impl<'a, F> CoverTreePrioritySearcher<'a, F>
where
    F: Float,
{
    pub fn new(tree: &'a CoverTree<F>) -> Self {
        let mut searcher = Self {
            tree,
            node_queue: BinaryHeap::new(),
            candidate_queue: CandidateHeap::new(),
            threshold: F::infinity(),
            skip_threshold: F::zero(),
        };
        searcher.reset();
        searcher
    }

    pub fn reset(&mut self) {
        self.node_queue.clear();
        self.candidate_queue.clear();
        self.threshold = F::infinity();
        self.skip_threshold = F::zero();

        if let Some(root) = self.tree.root.as_deref() {
            self.node_queue.push(NodeEntry {
                lower_bound: F::zero(),
                node: root,
                emit_center: true,
            });
        }
    }

    pub fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        self.reset();
        self.threshold = cutoff;
        self.skip_threshold = skip;
    }

    pub fn decrease_cutoff(&mut self, threshold: F) {
        debug_assert!(threshold <= self.threshold, "Thresholds must only decrease.");
        self.threshold = threshold;

        // Prune nodes that cannot contribute.
        while let Some(top) = self.node_queue.peek() {
            if top.lower_bound <= threshold {
                break;
            }
            self.node_queue.pop();
        }

        while let Some(candidate) = self.candidate_queue.peek() {
            if candidate.distance <= threshold {
                break;
            }
            self.candidate_queue.pop();
        }
    }

    pub fn all_lower_bound(&self) -> F {
        let node_bound = self.node_queue.peek().map_or(F::infinity(), |entry| entry.lower_bound);
        let candidate_bound =
            self.candidate_queue.peek().map_or(F::infinity(), |entry| entry.distance);
        node_bound.min(candidate_bound)
    }

    fn push_node_children(&mut self, node: &'a CoverTreeNode<F>, node_center_dist: F) {
        for child in &node.children {
            let child_lower = (node_center_dist - child.parent_dist).abs() - child.max_dist;
            if child_lower <= self.threshold {
                self.node_queue.push(NodeEntry {
                    lower_bound: child_lower,
                    node: child,
                    emit_center: child.center != node.center,
                });
            }
        }
    }

    fn expand_next_node<Q: DistanceSearch<F> + ?Sized>(&mut self, query: &Q) {
        if let Some(entry) = self.node_queue.pop() {
            if entry.lower_bound > self.threshold {
                return;
            }

            let node = entry.node;

            let d_center = query.query_distance(node.center);
            if entry.emit_center
                && d_center <= self.threshold
                && (self.skip_threshold == F::zero() || d_center >= self.skip_threshold)
            {
                self.candidate_queue.push(DistPair::new(d_center, node.center));
            }

            for &(idx, _) in node.singletons.iter() {
                let d = query.query_distance(idx);
                if d <= self.threshold
                    && (self.skip_threshold == F::zero() || d >= self.skip_threshold)
                {
                    self.candidate_queue.push(DistPair::new(d, idx));
                }
            }

            self.push_node_children(node, d_center);
        }
    }

    pub fn next<Q: DistanceSearch<F> + ?Sized>(&mut self, query: &Q) -> Option<DistPair<F>> {
        loop {
            let best_preview = self.candidate_queue.peek().map(|candidate| candidate.distance);
            let best_node_bound =
                self.node_queue.peek().map_or(F::infinity(), |entry| entry.lower_bound);

            if let Some(best_dist) = best_preview
                && best_dist <= best_node_bound
            {
                let candidate = self.candidate_queue.pop().unwrap();
                if candidate.distance > self.threshold {
                    return None;
                }
                if self.skip_threshold != F::zero() && candidate.distance < self.skip_threshold {
                    continue;
                }
                return Some(candidate);
            }

            if self.node_queue.is_empty() {
                if let Some(candidate) = self.candidate_queue.pop() {
                    if candidate.distance > self.threshold {
                        return None;
                    }
                    if self.skip_threshold != F::zero() && candidate.distance < self.skip_threshold {
                        continue;
                    }
                    return Some(candidate);
                }
                return None;
            }

            self.expand_next_node(query);
        }
    }

    pub fn next_with_filter<Q: DistanceSearch<F> + ?Sized, S>(
        &mut self, query: &Q, filter: &mut S,
    ) -> Option<DistPair<F>>
    where
        S: crate::api::SearchFilter,
    {
        while let Some(candidate) = self.next(query) {
            if !filter.skip_point(candidate.index) {
                return Some(candidate);
            }
        }
        None
    }
}

impl<'a, F, Q> PrioritySearcher<F, Q> for CoverTreePrioritySearcher<'a, F>
where
    F: Float,
    Q: DistanceSearch<F> + ?Sized,
{
    fn reset(&mut self) { CoverTreePrioritySearcher::reset(self); }

    fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        CoverTreePrioritySearcher::reset_with_limits(self, cutoff, skip);
    }

    fn next(&mut self, query: &Q) -> Option<DistPair<F>> {
        CoverTreePrioritySearcher::next(self, query)
    }

    fn next_with_filter<S2>(&mut self, query: &Q, filter: &mut S2) -> Option<DistPair<F>>
    where
        S2: crate::api::SearchFilter,
    {
        CoverTreePrioritySearcher::next_with_filter(self, query, filter)
    }

    fn all_lower_bound(&self) -> F { CoverTreePrioritySearcher::all_lower_bound(self) }

    fn decrease_cutoff(&mut self, threshold: F) {
        CoverTreePrioritySearcher::decrease_cutoff(self, threshold);
    }
}
