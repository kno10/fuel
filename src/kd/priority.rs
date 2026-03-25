use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::kd::KdTree;
use crate::{
    CoordinateSearch, DistPair, DistanceSearch, Float, PrioritySearcherFactory, SearchFilter,
};

#[derive(Debug, Clone, PartialEq)]
struct PriorityBranch<F> {
    mindist: F,
    left: usize,
    right: usize,
}

impl<F> PriorityBranch<F> {
    const fn new(mindist: F, left: usize, right: usize) -> Self { Self { mindist, left, right } }
}

impl<F: PartialEq> Eq for PriorityBranch<F> {}

impl<F: PartialOrd + PartialEq> PartialOrd for PriorityBranch<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<F: PartialOrd + PartialEq> Ord for PriorityBranch<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.mindist.partial_cmp(&self.mindist).unwrap_or(Ordering::Equal)
    }
}

/// Incremental priority searcher for the k-d-tree.
pub struct KdTreePrioritySearcher<'a, C, F>
where
    C: Float,
    F: Float,
{
    tree: &'a KdTree<C>,
    heap: BinaryHeap<PriorityBranch<F>>,
    threshold: F,
    skip_threshold: F,
    candidates: BinaryHeap<DistPair<F>>,
}

impl<'a, C, F> KdTreePrioritySearcher<'a, C, F>
where
    C: Float,
    F: Float,
{
    /// Create a new searcher for the given query.
    pub fn new(tree: &'a KdTree<C>) -> Self {
        let mut searcher = Self {
            tree,
            heap: BinaryHeap::new(),
            threshold: F::infinity(),
            skip_threshold: F::zero(),
            candidates: BinaryHeap::new(),
        };
        searcher.reset_queue();
        searcher
    }

    pub fn reset(&mut self) {
        self.reset_queue();
        self.threshold = F::infinity();
        self.skip_threshold = F::zero();
    }

    /// Reset the searcher and set upper/lower cutoff bounds.
    pub fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        self.reset_queue();
        self.threshold = cutoff;
        self.skip_threshold = skip;
    }

    fn reset_queue(&mut self) {
        self.heap.clear();
        self.candidates.clear();
        if !self.tree.is_empty() {
            self.heap.push(PriorityBranch::new(F::zero(), 0, self.tree.len()));
        }
    }

    fn push_branch(&mut self, mindist: F, left: usize, right: usize) {
        if left >= right || mindist > self.threshold {
            return;
        }
        self.heap.push(PriorityBranch::new(mindist, left, right));
    }

    /// Reduce the upper distance cutoff (values must only decrease).
    pub fn decrease_cutoff(&mut self, threshold: F) {
        assert!(threshold <= self.threshold, "cutoff must only decrease");
        self.threshold = threshold;
        while let Some(entry) = self.heap.peek() {
            if entry.mindist <= self.threshold {
                break;
            }
            self.heap.pop();
        }
        while let Some(candidate) = self.candidates.peek() {
            if candidate.distance <= self.threshold {
                break;
            }
            self.candidates.pop();
        }
    }

    /// Alias for `decrease_cutoff`.
    pub fn set_threshold(&mut self, threshold: F) { self.decrease_cutoff(threshold); }

    /// Lower bound for all remaining candidates.
    pub fn all_lower_bound(&self) -> F {
        let heap_bound = self.heap.peek().map_or(F::infinity(), |entry| entry.mindist);
        let candidate_bound =
            self.candidates.peek().map_or(F::infinity(), |candidate| candidate.distance);
        self.skip_threshold.max(heap_bound.min(candidate_bound))
    }

    fn next_internal<Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized>(
        &mut self, query: &Q,
    ) -> Option<DistPair<F>> {
        loop {
            if let Some(candidate) = self.candidates.peek()
                && self.heap.peek().is_none_or(|entry| entry.mindist >= candidate.distance)
            {
                let cand = self.candidates.pop();
                if let Some(c) = &cand && c.distance < self.skip_threshold {
                    continue;
                }
                return cand;
            }

            let branch = match self.heap.pop() {
                Some(branch) => branch,
                None => {
                    if let Some(cand) = self.candidates.pop() {
                        if cand.distance < self.skip_threshold {
                            continue;
                        }
                        return Some(cand);
                    }
                    return None;
                }
            };

            if branch.mindist > self.threshold {
                continue;
            }

            let node_idx = usize::midpoint(branch.left, branch.right);
            let point_idx = self.tree.points[node_idx];
            let axis = self.tree.split_axes[node_idx];
            let split_value = self.tree.split_values[node_idx];
            let diff = query.query_coordinate(axis) - split_value;
            let axis_dist = query.delta_to_distance(diff);

            if branch.left < node_idx {
                let left_mindist = if diff > C::zero() {
                    query.combine_axis_distances(branch.mindist, axis_dist)
                } else {
                    branch.mindist
                };
                self.push_branch(left_mindist, branch.left, node_idx);
            }

            if node_idx + 1 < branch.right {
                let right_mindist = if diff < C::zero() {
                    query.combine_axis_distances(branch.mindist, axis_dist)
                } else {
                    branch.mindist
                };
                self.push_branch(right_mindist, node_idx + 1, branch.right);
            }

            let dist = query.query_distance(point_idx);
            if dist <= self.threshold {
                self.candidates.push(DistPair::new(dist, point_idx));
            }
        }
    }
}

impl<C: Float, F: Float, Q> PrioritySearcherFactory<F, Q> for KdTree<C>
where
    Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
{
    type Searcher<'a>
        = KdTreePrioritySearcher<'a, C, F>
    where
        C: 'a,
        F: 'a,
        Q: 'a;

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a,
    {
        KdTreePrioritySearcher::new(self)
    }
}

impl<'a, C, F, Q> crate::PrioritySearcher<F, Q> for KdTreePrioritySearcher<'a, C, F>
where
    C: Float,
    F: Float,
    Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
{
    fn reset(&mut self) { KdTreePrioritySearcher::reset(self); }

    fn reset_with_limits(&mut self, cutoff: F, skip: F) { self.reset_with_limits(cutoff, skip); }

    fn next(&mut self, query: &Q) -> Option<crate::DistPair<F>> { self.next_internal(query) }

    fn next_with_filter<S>(&mut self, query: &Q, filter: &mut S) -> Option<DistPair<F>>
    where
        S: SearchFilter,
    {
        loop {
            let cand = self.next_internal(query)?;
            if !filter.skip_point(cand.index) {
                return Some(cand);
            }
        }
    }

    fn all_lower_bound(&self) -> F { self.all_lower_bound() }

    fn decrease_cutoff(&mut self, threshold: F) { self.decrease_cutoff(threshold); }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::api::{DistanceData, PrioritySearcher};
    use crate::distance::{EuclideanDistance, SquaredEuclideanDistance};
    use crate::kd::MaxVarianceSplit;
    use crate::{CoordinateQuery, KnnSearch, TableWithDistance};

    fn sample_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]]
    }

    #[test]
    fn priority_search_produces_knn_in_order() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit);

        let query = data.query().with_coordinates(&points[0]);
        let mut searcher = crate::kd::priority::KdTreePrioritySearcher::new(&tree);
        let neighbors: Vec<_> = std::iter::from_fn(|| searcher.next(&query)).take(3).collect();

        assert!(neighbors.windows(2).all(|pair| pair[0].distance <= pair[1].distance));

        let expected: Vec<crate::DistPair<f64>> = tree.search_knn(&query, 3);
        assert_eq!(neighbors.len(), expected.len());

        let neighbor_ids: HashSet<_> = neighbors.iter().map(|cand| cand.index).collect();
        let expected_ids: HashSet<_> = expected.iter().map(|neighbor| neighbor.index).collect();
        assert_eq!(neighbor_ids, expected_ids);

        for (cand, neighbor) in neighbors.iter().zip(expected.iter()) {
            let diff: f64 = cand.distance - neighbor.distance;
            assert!(diff.abs() <= 1e-9);
        }
    }

    #[test]
    fn priority_search_can_decrease_cutoff() {
        let points = sample_points();
        let data: TableWithDistance<'_, f64, Vec<f64>, EuclideanDistance, f64> =
            TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit);
        let mut searcher = crate::kd::priority::KdTreePrioritySearcher::new(&tree);
        searcher.decrease_cutoff(0.5);
        assert!(searcher.all_lower_bound() <= 0.5);
    }
}
