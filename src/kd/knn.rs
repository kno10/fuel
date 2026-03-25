use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::kd::KdTree;
use crate::{CoordinateSearch, DistPair, DistanceSearch, Float, KnnSearch};

impl<C, F, Q> KnnSearch<F, Q> for KdTree<C>
where
    C: Float,
    F: Float + 'static,
    Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
{
    /// k nearest neighbors in order of increasing distance.
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> {
        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        let mut heap: BinaryHeap<Reverse<DistPair<F>>> = BinaryHeap::with_capacity(k + 1);
        self.search_knn_recursive(query, k, 0, self.points.len(), &mut heap, F::zero());

        let mut result: Vec<DistPair<F>> = heap.into_vec().into_iter().map(|r| r.0).collect();
        result.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }
}

impl<C> KdTree<C>
where
    C: Float,
{
    fn search_knn_recursive<F, Q>(
        &self, query: &Q, k: usize, left: usize, right: usize,
        heap: &mut BinaryHeap<Reverse<DistPair<F>>>, lower_bound: F,
    ) -> F
    where
        F: Float,
        Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
    {
        if left >= right {
            return self.tau(heap, k);
        }

        let mut tau = self.tau(heap, k);
        if lower_bound > tau {
            return tau;
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx];
        let dist = query.query_distance(point_idx);

        if heap.len() < k {
            heap.push(Reverse(DistPair::new(dist, point_idx)));
        } else if dist < heap.peek().unwrap().0.distance {
            heap.pop();
            heap.push(Reverse(DistPair::new(dist, point_idx)));
        }

        tau = self.tau(heap, k);
        let axis = self.split_axes[node_idx];
        let split = self.split_values[node_idx];

        let delta = query.query_coordinate(axis) - split;
        let plane_dist = query.delta_to_distance(delta);

        let (first, second) = if delta <= C::zero() {
            (
                (left, node_idx, lower_bound),
                (node_idx + 1, right, query.combine_axis_distances(lower_bound, plane_dist)),
            )
        } else {
            (
                (node_idx + 1, right, lower_bound),
                (left, node_idx, query.combine_axis_distances(lower_bound, plane_dist)),
            )
        };

        if first.0 < first.1 {
            tau = self.search_knn_recursive(query, k, first.0, first.1, heap, first.2);
        }
        if second.0 < second.1 && second.2 <= tau {
            tau = self.search_knn_recursive(query, k, second.0, second.1, heap, second.2);
        }

        tau
    }

    fn tau<F: Float>(&self, heap: &BinaryHeap<Reverse<DistPair<F>>>, k: usize) -> F {
        if heap.len() < k { F::infinity() } else { heap.peek().unwrap().0.distance }
    }
}
