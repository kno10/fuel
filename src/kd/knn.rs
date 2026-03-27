use crate::kd::KdTree;
use crate::{CoordinateSearch, DistPair, DistanceSearch, Float, KNNHeap, KnnSearch};

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

        let mut heap: KNNHeap<F> = KNNHeap::new(k);
        self.search_knn_recursive(query, 0, self.points.len(), &mut heap, F::zero());

        heap.into_vec()
    }
}

impl<C> KdTree<C>
where
    C: Float,
{
    fn search_knn_recursive<F, Q>(
        &self, query: &Q, left: usize, right: usize, heap: &mut KNNHeap<F>,
        lower_bound: F,
    ) -> F
    where
        F: Float,
        Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
    {
        if left >= right {
            return self.tau(heap);
        }

        let mut tau = self.tau(heap);
        if lower_bound > tau {
            return tau;
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx];
        let dist = query.query_distance(point_idx);

        tau = heap.insert(DistPair::new(dist, point_idx));
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
            tau = self.search_knn_recursive(query, first.0, first.1, heap, first.2);
        }
        if second.0 < second.1 && second.2 <= tau {
            tau = self.search_knn_recursive(query, second.0, second.1, heap, second.2);
        }

        tau
    }

    fn tau<F: Float>(&self, heap: &KNNHeap<F>) -> F { heap.k_distance() }
}
