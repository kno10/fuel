use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::search::covertree::CoverTree;
use crate::{CandidateHeap, DistPair, DistanceSearch, Float, PrioritySearcher};

#[derive(Debug, Clone, Copy)]
struct NodeEntry<F>
where
    F: Float,
{
    lower_bound: F,
    node_idx: u32,
    emit_center: bool,
    center_dist: F,
}

impl<F: Float> PartialEq for NodeEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound
            .partial_cmp(&other.lower_bound)
            .map(|o| o == Ordering::Equal)
            .unwrap_or(false)
    }
}

impl<F: Float> Eq for NodeEntry<F> {}

impl<F: Float> PartialOrd for NodeEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<F: Float> Ord for NodeEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower_bound.partial_cmp(&self.lower_bound).unwrap_or(Ordering::Equal)
    }
}

pub struct CoverTreePrioritySearcher<'a, F>
where
    F: Float,
{
    tree: &'a CoverTree<F>,
    node_queue: BinaryHeap<NodeEntry<F>>,
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

        self.node_queue.push(NodeEntry {
            lower_bound: F::zero(),
            node_idx: 0,
            emit_center: true,
            center_dist: F::infinity(),
        });
    }

    pub fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        self.reset();
        self.threshold = cutoff;
        self.skip_threshold = skip;
    }

    pub fn decrease_cutoff(&mut self, threshold: F) {
        debug_assert!(threshold <= self.threshold, "Thresholds must only decrease.");
        self.threshold = threshold;

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

    fn expand_next_node<Q: DistanceSearch<F> + ?Sized>(&mut self, query: &Q) {
        if let Some(entry) = self.node_queue.pop() {
            if entry.lower_bound > self.threshold {
                return;
            }

            let node = &self.tree.nodes[entry.node_idx as usize];
            let d_center = if entry.center_dist.is_infinite() {
                query.query_distance(node.center)
            } else {
                entry.center_dist
            };

            if entry.emit_center
                && d_center <= self.threshold
                && (self.skip_threshold == F::zero() || d_center >= self.skip_threshold)
            {
                self.candidate_queue.push(DistPair::new(d_center, node.center));
            }

            for singleton in node.singletons.iter() {
                let idx = singleton.index;
                let d = query.query_distance(idx);
                if d <= self.threshold
                    && (self.skip_threshold == F::zero() || d >= self.skip_threshold)
                {
                    self.candidate_queue.push(DistPair::new(d, idx));
                }
            }

            for &child_idx in &node.children {
                let child = &self.tree.nodes[child_idx as usize];

                // Parent-child pruning check, same as range query.
                let parent_child_bound = (d_center - child.parent_dist).abs();
                if parent_child_bound - child.max_dist > self.threshold {
                    continue;
                }

                let d_child = query.query_distance(child.center);
                let child_lower = d_child - child.max_dist;
                if child_lower <= self.threshold {
                    self.node_queue.push(NodeEntry {
                        lower_bound: child_lower,
                        node_idx: child_idx,
                        emit_center: child.center != node.center,
                        center_dist: d_child,
                    });
                }
            }
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
                    if self.skip_threshold != F::zero() && candidate.distance < self.skip_threshold
                    {
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

#[cfg(test)]
mod tests {
    use crate::distance::SquaredEuclidean;
    use crate::search::covertree::{CoverTree, CoverTreePrioritySearcher};
    use crate::{CoordinateQuery, DistPair, DistanceData, TableWithDistance};

    fn sample_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]]
    }

    #[test]
    fn cover_tree_priority_search_can_decrease_cutoff() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = CoverTree::new(&data, 1.3, 0);
        let query = data.query().with_coordinates(&points[0]);
        let mut searcher = CoverTreePrioritySearcher::new(&tree);

        let first = searcher.next(&query).expect("should return first neighbor");
        assert_eq!(first.index, 0);

        searcher.decrease_cutoff(0.5);
        assert!(searcher.next(&query).is_none());
    }

    #[test]
    fn cover_tree_priority_order_matches_knn() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = CoverTree::new(&data, 1.3, 0);
        let query = data.query().with_coordinates(&points[0]);

        let knn: Vec<DistPair<f64>> = tree.search_knn(&query, 4);
        let mut searcher = CoverTreePrioritySearcher::new(&tree);
        let mut ks: Vec<DistPair<f64>> = Vec::new();
        for _ in 0..4 {
            if let Some(neighbor) = searcher.next(&query) {
                ks.push(neighbor);
            }
        }

        assert!(ks.len() >= 4);
        for (a, b) in ks.iter().zip(knn.iter()) {
            assert_eq!(a.index, b.index);
            let diff: f64 = (a.distance - b.distance).abs();
            assert!(diff < 1e-6);
        }
    }
}
