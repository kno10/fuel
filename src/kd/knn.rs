use std::cmp::Reverse;
use std::collections::BinaryHeap;

use num_traits::Float;

use crate::api::VectorData;
use crate::distance::PartialDistance;

use crate::DistPair;
use super::KdTree;

impl<F, M> KdTree<F, M>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
{
    /// k nearest neighbors in order of increasing distance.
    pub fn search_knn<P>(&self, data: &P, query: &[F], k: usize) -> Vec<DistPair<F>>
    where
        P: VectorData<F> + ?Sized,
    {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        self.check_query(query);
        let mut heap: BinaryHeap<Reverse<DistPair<F>>> = BinaryHeap::with_capacity(k + 1);
        self.search_knn_recursive(data, query, k, 0, self.points.len(), &mut heap, F::zero());

        let mut result: Vec<DistPair<F>> = heap
            .into_vec()
            .into_iter()
            .map(|r| r.0)
            .collect();
        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    fn search_knn_recursive<P>(
        &self,
        data: &P,
        query: &[F],
        k: usize,
        left: usize,
        right: usize,
        heap: &mut BinaryHeap<Reverse<DistPair<F>>>,
        lower_bound: F,
    ) -> F
    where
        P: VectorData<F> + ?Sized,
    {
        if left >= right {
            return self.tau(heap, k);
        }

        // Prune subtree early if its lower bound is already worse than the best
        // k-th distance in the current result set.
        let mut tau = self.tau(heap, k);
        if lower_bound > tau {
            return tau;
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx];
        let dist = self.metric.distance(query, data.point(point_idx));

        if heap.len() < k {
            heap.push(Reverse(DistPair::new(dist, point_idx)));
        } else if dist < heap.peek().unwrap().0.distance {
            heap.pop();
            heap.push(Reverse(DistPair::new(dist, point_idx)));
        }

        tau = self.tau(heap, k);
        let axis = self.split_axes[node_idx];
        let split = self.split_values[node_idx];
        let diff = query[axis] - split;
        let plane_dist = self.metric.axis_distance(diff);

        // Determine traversal order: first visit the side containing the query.
        let (first, second) = if diff <= F::zero() {
            (
                (left, node_idx, lower_bound),
                (
                    node_idx + 1,
                    right,
                    self.metric.combine_axis_distances(lower_bound, plane_dist),
                ),
            )
        } else {
            (
                (node_idx + 1, right, lower_bound),
                (
                    left,
                    node_idx,
                    self.metric.combine_axis_distances(lower_bound, plane_dist),
                ),
            )
        };

        if first.0 < first.1 {
            tau = self.search_knn_recursive(
                data,
                query,
                k,
                first.0,
                first.1,
                heap,
                first.2,
            );
        }
        if second.0 < second.1 && second.2 <= tau {
            tau = self.search_knn_recursive(
                data,
                query,
                k,
                second.0,
                second.1,
                heap,
                second.2,
            );
        }

        tau
    }

    fn tau(&self, heap: &BinaryHeap<Reverse<DistPair<F>>>, k: usize) -> F {
        if heap.len() < k {
            F::infinity()
        } else {
            heap.peek().unwrap().0.distance
        }
    }
}
