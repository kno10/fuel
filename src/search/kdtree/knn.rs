use crate::search::kdtree::KdTree;
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

        if self.is_empty() {
            return Vec::new();
        }

        let mut heap: KNNHeap<F> = KNNHeap::new(k);
        let mut axis_bounds = vec![F::zero(); query.dims()];
        self.search_knn_recursive(
            query,
            0,
            self.points.len(),
            &mut heap,
            F::zero(),
            &mut axis_bounds,
        );

        heap.into_vec()
    }
}

impl<C> KdTree<C>
where
    C: Float,
{
    fn search_knn_recursive<F, Q>(
        &self, query: &Q, left: usize, right: usize, heap: &mut KNNHeap<F>, lower_bound: F,
        axis_bounds: &mut [F],
    ) -> F
    where
        F: Float,
        Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
    {
        if left >= right {
            return self.tau(heap);
        }

        let mut tau = self.tau(heap);
        let mut tau_bound = query.distance_to_range_bound(tau);
        if lower_bound > tau_bound {
            return tau;
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx] as usize;
        let dist = query.query_distance(point_idx);

        tau = heap.insert(DistPair::new(dist, point_idx));
        tau_bound = query.distance_to_range_bound(tau);
        let axis = self.split_axes[node_idx] as usize;
        let split = self.split_values[node_idx];

        let delta = query.query_coordinate(axis) - split;
        let plane_dist = query.delta_to_distance(delta);

        let (first, second) = if delta <= C::zero() {
            ((left, node_idx, lower_bound), (node_idx + 1, right, lower_bound))
        } else {
            ((node_idx + 1, right, lower_bound), (left, node_idx, lower_bound))
        };

        if first.0 < first.1 {
            tau = self.search_knn_recursive(query, first.0, first.1, heap, first.2, axis_bounds);
            tau_bound = query.distance_to_range_bound(tau);
        }

        if second.0 < second.1 {
            let old_axis_bound = axis_bounds[axis];
            let new_axis_bound =
                if plane_dist > old_axis_bound { plane_dist } else { old_axis_bound };
            axis_bounds[axis] = new_axis_bound;

            let second_lower_bound = query.replace_axis_distance(
                lower_bound,
                axis,
                old_axis_bound,
                new_axis_bound,
                axis_bounds,
            );

            if second_lower_bound <= tau_bound {
                tau = self.search_knn_recursive(
                    query,
                    second.0,
                    second.1,
                    heap,
                    second_lower_bound,
                    axis_bounds,
                );
            }

            axis_bounds[axis] = old_axis_bound;
        }

        tau
    }

    fn tau<F: Float>(&self, heap: &KNNHeap<F>) -> F { heap.k_distance() }
}
