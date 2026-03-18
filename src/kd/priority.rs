use std::{cmp::Ordering, collections::BinaryHeap};

use num_traits::Float;

use crate::api::VectorData;
use crate::distance::PartialDistance;
use crate::DistPair;

use super::KdTree;

#[derive(Debug, Clone, PartialEq)]
struct PriorityBranch<F> {
    mindist: F,
    left: usize,
    right: usize,
}

impl<F> PriorityBranch<F> {
    const fn new(mindist: F, left: usize, right: usize) -> Self {
        Self {
            mindist,
            left,
            right,
        }
    }
}

impl<F: PartialEq> Eq for PriorityBranch<F> {}

impl<F: PartialOrd + PartialEq> PartialOrd for PriorityBranch<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: PartialOrd + PartialEq> Ord for PriorityBranch<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .mindist
            .partial_cmp(&self.mindist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Incremental priority searcher for the k-d-tree.
pub struct KdTreePrioritySearcher<'a, F, M, D, P>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
    D: PartialDistance<F> + Clone,
    P: VectorData<F> + ?Sized,
{
    tree: &'a KdTree<F, M>,
    data: &'a P,
    query: Vec<F>,
    heap: BinaryHeap<PriorityBranch<F>>,
    metric: D,
    threshold: F,
    candidates: BinaryHeap<DistPair<F>>,
}

impl<'a, F, M, D, P> KdTreePrioritySearcher<'a, F, M, D, P>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
    D: PartialDistance<F> + Clone,
    P: VectorData<F> + ?Sized,
{
    /// Create a new searcher for the given query.
    pub fn new(tree: &'a KdTree<F, M>, data: &'a P, query: &[F], metric: D) -> Self {
        tree.check_query(query);
        let mut searcher = Self {
            tree,
            data,
            query: query.to_vec(),
            heap: BinaryHeap::new(),
            metric,
            threshold: F::infinity(),
            candidates: BinaryHeap::new(),
        };
        searcher.reset_queue();
        searcher
    }

    /// Restart the searcher with a different query.
    pub fn search(&mut self, query: &[F]) -> &mut Self {
        self.tree.check_query(query);
        self.query.clear();
        self.query.extend_from_slice(query);
        self.reset_queue();
        self.threshold = F::infinity();
        self
    }

    /// Reset the searcher while keeping the current query.
    pub fn reset(&mut self) {
        self.reset_queue();
        self.threshold = F::infinity();
    }

    fn reset_queue(&mut self) {
        self.heap.clear();
        self.candidates.clear();
        if !self.tree.is_empty() {
            self.heap
                .push(PriorityBranch::new(F::zero(), 0, self.tree.len()));
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
    pub fn set_threshold(&mut self, threshold: F) {
        self.decrease_cutoff(threshold);
    }

    /// Lower bound for all remaining candidates.
    pub fn all_lower_bound(&self) -> F {
        let heap_bound = self
            .heap
            .peek()
            .map_or(F::infinity(), |entry| entry.mindist);
        let candidate_bound = self
            .candidates
            .peek()
            .map_or(F::infinity(), |candidate| candidate.distance);
        heap_bound.min(candidate_bound)
    }
}

impl<'a, F, M, D, P> Iterator for KdTreePrioritySearcher<'a, F, M, D, P>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
    D: PartialDistance<F> + Clone,
    P: VectorData<F> + ?Sized,
{
    type Item = DistPair<F>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(candidate) = self.candidates.peek()
                && self
                    .heap
                    .peek()
                    .is_none_or(|entry| entry.mindist >= candidate.distance)
            {
                return self.candidates.pop();
            }

            let branch = match self.heap.pop() {
                Some(branch) => branch,
                None => return self.candidates.pop(),
            };

            if branch.mindist > self.threshold {
                continue;
            }

            let node_idx = usize::midpoint(branch.left, branch.right);
            let point_idx = self.tree.points[node_idx];
            let axis = self.tree.split_axes[node_idx];
            let split_value = self.tree.split_values[node_idx];
            let diff = self.query[axis] - split_value;
            let axis_dist = self.metric.axis_distance(diff);

            if branch.left < node_idx {
                let left_mindist = if diff > F::zero() {
                    self.metric.combine_axis_distances(branch.mindist, axis_dist)
                } else {
                    branch.mindist
                };
                self.push_branch(left_mindist, branch.left, node_idx);
            }

            if node_idx + 1 < branch.right {
                let right_mindist = if diff < F::zero() {
                    self.metric.combine_axis_distances(branch.mindist, axis_dist)
                } else {
                    branch.mindist
                };
                self.push_branch(right_mindist, node_idx + 1, branch.right);
            }

            let dist = self.metric.distance(&self.query, self.data.point(point_idx));
            if dist <= self.threshold {
                self.candidates.push(DistPair::new(dist, point_idx));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        TableWithDistance,
        distance::{EuclideanDistance, SquaredEuclideanDistance},
        kd::MaxVarianceSplit,
    };
    use std::collections::HashSet;

    use super::*;

    fn sample_points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ]
    }

    #[test]
    fn priority_search_produces_knn_in_order() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit, SquaredEuclideanDistance);

        let mut searcher = tree.priority_searcher(&data, &points[0]);
        let neighbors: Vec<_> = searcher.by_ref().take(3).collect();

        assert!(
            neighbors
                .windows(2)
                .all(|pair| pair[0].distance <= pair[1].distance)
        );

        let expected = tree.search_knn(&data, &points[0], 3);
        assert_eq!(neighbors.len(), expected.len());

        let neighbor_ids: HashSet<_> = neighbors.iter().map(|cand| cand.index).collect();
        let expected_ids: HashSet<_> = expected.iter().map(|neighbor| neighbor.index).collect();
        assert_eq!(neighbor_ids, expected_ids);

        for (cand, neighbor) in neighbors.iter().zip(expected.iter()) {
            let diff = cand.distance - neighbor.distance;
            assert!(diff.abs() <= 1e-9);
        }
    }

    #[test]
    fn priority_search_can_decrease_cutoff() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit, EuclideanDistance);
        let mut searcher = tree.priority_searcher(&data, &points[0]);
        searcher.decrease_cutoff(0.5);
        assert!(searcher.all_lower_bound() <= 0.5);
    }
}
