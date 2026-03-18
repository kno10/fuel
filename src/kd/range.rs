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
    /// Range search returning all points within `radius` of the query point.
    pub fn search_range<P>(&self, data: &P, query: &[F], radius: F) -> Vec<DistPair<F>>
    where
        P: VectorData<F> + ?Sized,
    {
        if self.is_empty() || radius.is_sign_negative() {
            return Vec::new();
        }

        self.check_query(query);
        let mut result = Vec::new();
        self.search_range_recursive(data, query, radius, 0, self.points.len(), &mut result, F::zero());
        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        result
    }

    fn search_range_recursive<P>(
        &self,
        data: &P,
        query: &[F],
        radius: F,
        left: usize,
        right: usize,
        result: &mut Vec<DistPair<F>>,
        lower_bound: F,
    ) where
        P: VectorData<F> + ?Sized,
    {
        if left >= right {
            return;
        }

        if lower_bound > radius {
            return;
        }

        let node_idx = usize::midpoint(left, right);
        let point_idx = self.points[node_idx];
        let dist = self.metric.distance(query, data.point(point_idx));

        if dist <= radius {
            result.push(DistPair::new(dist, point_idx));
        }

        let axis = self.split_axes[node_idx];
        let split = self.split_values[node_idx];
        let diff = query[axis] - split;
        let plane_dist = self.metric.axis_distance(diff);

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
            self.search_range_recursive(
                data,
                query,
                radius,
                first.0,
                first.1,
                result,
                first.2,
            );
        }
        if second.0 < second.1 && second.2 <= radius {
            self.search_range_recursive(
                data,
                query,
                radius,
                second.0,
                second.1,
                result,
                second.2,
            );
        }
    }
}
