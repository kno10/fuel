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

        let mut result = Vec::new();
        self.search_range_recursive(query, radius, 0, self.points.len(), &mut result, F::zero());
        // TODO: make sorting optional
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
    fn search_range_recursive<F, Q>(
        &self, query: &Q, radius: F, left: usize, right: usize, result: &mut Vec<DistPair<F>>,
        lower_bound: F,
    ) where
        F: Float,
        Q: DistanceSearch<F> + CoordinateSearch<C, F> + ?Sized,
    {
        if left >= right {
            return;
        }

        if lower_bound > radius {
            return;
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx];
        let dist = query.query_distance(point_idx);

        if dist <= radius {
            result.push(DistPair::new(dist, point_idx));
        }

        let axis = self.split_axes[node_idx];
        let split = self.split_values[node_idx];
        let diff = query.query_coordinate(axis) - split;
        let plane_dist = query.delta_to_distance(diff);

        let (first, second) = if diff <= C::zero() {
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
            self.search_range_recursive(query, radius, first.0, first.1, result, first.2);
        }
        if second.0 < second.1 && second.2 <= radius {
            self.search_range_recursive(query, radius, second.0, second.1, result, second.2);
        }
    }
}
