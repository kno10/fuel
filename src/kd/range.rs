use crate::kd::KdTree;
use crate::{CoordinateSearch, DistPair, DistanceSearch, Float, RangeSearch};

impl<C, F, Q> RangeSearch<F, Q> for KdTree<C>
where
    C: Float,
    F: Float + 'static,
    Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
{
    /// Range search returning all points within `radius` of the query point.
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        if self.is_empty() || radius.is_sign_negative() {
            return Vec::new();
        }

        let mut axis_bounds = vec![F::zero(); query.dims()];
        let mut result = Vec::new();
        let bound_radius = query.distance_to_range_bound(radius);
        self.search_range_recursive(
            query,
            radius,
            bound_radius,
            0,
            self.points.len(),
            &mut result,
            &mut axis_bounds,
        );
        // TODO: make sorting optional
        result.sort();
        result
    }
}

impl<C> KdTree<C>
where
    C: Float,
{
    fn search_range_recursive<F, Q>(
        &self, query: &Q, radius: F, radius_bound: F, left: usize, right: usize,
        result: &mut Vec<DistPair<F>>, axis_bounds: &mut [F],
    ) where
        F: Float,
        Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
    {
        if left >= right {
            return;
        }

        let mut lower_bound = F::zero();
        for (axis, &axis_dist) in axis_bounds.iter().enumerate() {
            lower_bound =
                query.replace_axis_distance(lower_bound, axis, F::zero(), axis_dist, axis_bounds);
        }

        if lower_bound > radius_bound {
            return;
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx] as usize;
        let dist = query.query_distance(point_idx);

        if dist <= radius {
            result.push(DistPair::new(dist, point_idx));
        }

        let axis = self.split_axes[node_idx] as usize;
        let split = self.split_values[node_idx];
        let diff = query.query_coordinate(axis) - split;
        let plane_dist = query.delta_to_distance(diff);

        let (first_range, second_range) = if diff <= C::zero() {
            ((left, node_idx), (node_idx + 1, right))
        } else {
            ((node_idx + 1, right), (left, node_idx))
        };

        if first_range.0 < first_range.1 {
            self.search_range_recursive(
                query,
                radius,
                radius_bound,
                first_range.0,
                first_range.1,
                result,
                axis_bounds,
            );
        }

        if second_range.0 < second_range.1 {
            let old_axis_bound = axis_bounds[axis];
            let new_axis_bound =
                if plane_dist > old_axis_bound { plane_dist } else { old_axis_bound };
            axis_bounds[axis] = new_axis_bound;

            let axis_lower_bound = query.replace_axis_distance(
                lower_bound,
                axis,
                old_axis_bound,
                new_axis_bound,
                axis_bounds,
            );

            if axis_lower_bound <= radius_bound {
                self.search_range_recursive(
                    query,
                    radius,
                    radius_bound,
                    second_range.0,
                    second_range.1,
                    result,
                    axis_bounds,
                );
            }

            axis_bounds[axis] = old_axis_bound;
        }
    }
}
