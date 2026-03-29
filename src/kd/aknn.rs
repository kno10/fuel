use crate::kd::KdTree;
use crate::{ApproxKnnSearch, CoordinateSearch, DistPair, DistanceSearch, Float, KNNHeap};

impl<C, F, Q> ApproxKnnSearch<F, Q> for KdTree<C>
where
    C: Float,
    F: Float + 'static,
    Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
{
    fn search_aknn(&self, query: &Q, k: usize, rate: f32) -> Vec<DistPair<F>> {
        if k == 0 || self.is_empty() || rate <= 0.0 || !rate.is_finite() {
            return Vec::new();
        }

        let max_dists = (rate * (self.points.len() as f32)).ceil() as usize;
        let max_dists = max_dists.min(self.points.len());
        if max_dists == 0 {
            return Vec::new();
        }

        let mut heap: KNNHeap<F> = KNNHeap::new(k);
        let mut axis_bounds = vec![F::zero(); query.dims()];
        let mut dist_count = 0;

        self.search_aknn_recursive(
            query,
            0,
            self.points.len(),
            &mut heap,
            F::zero(),
            &mut axis_bounds,
            &mut dist_count,
            max_dists,
        );

        heap.into_vec()
    }
}

impl<C> KdTree<C>
where
    C: Float,
{
    fn search_aknn_recursive<F, Q>(
        &self, query: &Q, left: usize, right: usize, heap: &mut KNNHeap<F>, lower_bound: F,
        axis_bounds: &mut [F], dist_count: &mut usize, max_dists: usize,
    ) -> (F, bool)
    where
        F: Float,
        Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
    {
        if left >= right || *dist_count >= max_dists {
            return (heap.k_distance(), *dist_count >= max_dists);
        }

        let mut tau = heap.k_distance();
        let mut tau_bound = query.distance_to_range_bound(tau);
        if lower_bound > tau_bound {
            return (tau, false);
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx] as usize;

        if *dist_count >= max_dists {
            return (tau, true);
        }

        let dist = query.query_distance(point_idx);
        *dist_count += 1;

        tau = heap.insert(DistPair::new(dist, point_idx));
        tau_bound = query.distance_to_range_bound(tau);

        if *dist_count >= max_dists {
            return (tau, true);
        }

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
            let (new_tau, done) = self.search_aknn_recursive(
                query,
                first.0,
                first.1,
                heap,
                first.2,
                axis_bounds,
                dist_count,
                max_dists,
            );
            tau = new_tau;
            tau_bound = query.distance_to_range_bound(tau);
            if done {
                return (tau, true);
            }
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
                let (new_tau, done) = self.search_aknn_recursive(
                    query,
                    second.0,
                    second.1,
                    heap,
                    second_lower_bound,
                    axis_bounds,
                    dist_count,
                    max_dists,
                );
                tau = new_tau;
                if done {
                    return (tau, true);
                }
            }

            axis_bounds[axis] = old_axis_bound;
        }

        (tau, false)
    }
}
