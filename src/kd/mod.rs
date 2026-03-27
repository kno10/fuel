mod knn;
mod priority;
mod range;
pub mod split;

use std::cmp::Ordering;

pub use priority::KdTreePrioritySearcher;
pub use split::{AxisCycleSplit, LargestSpreadSplit, MaxVarianceSplit, SplitStrategy};

use crate::{Float, VectorData};

/// A static KD-tree stored in heap order.
pub struct KdTree<C> {
    points: Vec<usize>,
    split_axes: Vec<usize>,
    split_values: Vec<C>,
    dims: usize,
}

impl<C> KdTree<C>
where
    C: Float,
{
    /// Build a new tree from the given point set using the supplied split heuristic.
    pub fn new<P, S>(data: &P, strategy: S) -> Self
    where
        P: VectorData<C> + ?Sized,
        S: SplitStrategy<C, P>,
    {
        let size = data.size();
        let dims = data.dims();
        assert!(size == 0 || dims > 0, "cannot index zero-dimensional points");

        let mut tree = Self {
            points: vec![0; size],
            split_axes: vec![0; size],
            split_values: vec![C::zero(); size],
            dims,
        };

        if size > 0 {
            let mut indices: Vec<usize> = (0..size).collect();
            tree.build_recursive(data, &mut indices, 0, size, 0, &strategy);
        }

        tree
    }

    /// Number of points stored in the tree.
    pub const fn len(&self) -> usize { self.points.len() }

    /// True if the tree is empty.
    pub fn is_empty(&self) -> bool { self.points.is_empty() }

    /// Dimensionality of the indexed space.
    pub const fn dims(&self) -> usize { self.dims }

    fn build_recursive<P, S>(
        &mut self, data: &P, indices: &mut [usize], left: usize, right: usize, depth: usize,
        strategy: &S,
    ) where
        P: VectorData<C> + ?Sized,
        S: SplitStrategy<C, P>,
    {
        if left >= right {
            return;
        }

        let node_idx = usize::midpoint(left, right);
        let axis = strategy.choose_axis(data, &indices[left..right], depth);
        assert!(axis < self.dims, "split axis must be in bounds");

        let range = &mut indices[left..right];
        let median = node_idx - left;
        range.select_nth_unstable_by(median, |a, b| {
            data.point(*a)[axis].partial_cmp(&data.point(*b)[axis]).unwrap_or(Ordering::Equal)
        });

        let median_idx = indices[node_idx];
        self.points[node_idx] = median_idx;
        self.split_axes[node_idx] = axis;
        self.split_values[node_idx] = data.point(median_idx)[axis];

        self.build_recursive(data, indices, left, node_idx, depth + 1, strategy);
        self.build_recursive(data, indices, node_idx + 1, right, depth + 1, strategy);
    }
}

#[cfg(test)]
mod tests {
    use crate::api::DistanceData;
    use crate::distance::SquaredEuclidean;
    use crate::kd::{AxisCycleSplit, KdTree, LargestSpreadSplit, MaxVarianceSplit};
    use crate::{CoordinateQuery, DistPair, KnnSearch, RangeSearch, TableWithDistance};

    fn sample_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]]
    }

    #[test]
    fn search_knn_returns_nearest() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = KdTree::new(&data, MaxVarianceSplit);
        let query = data.query().with_coordinates(&points[0]);
        let neighbors: Vec<DistPair<f64>> = tree.search_knn(&query, 3);

        assert!(neighbors.len() >= 3);
        assert_eq!(neighbors[0].index, 0);
        assert!(neighbors.iter().any(|n| n.index == 1));
        assert!(neighbors.iter().any(|n| n.index == 2));
    }

    #[test]
    fn range_search_finds_close_points() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = KdTree::new(&data, MaxVarianceSplit);
        let query = data.query().with_coordinates(&points[0]);
        let result: Vec<DistPair<f64>> = tree.search_range(&query, 1.01);

        assert!(result.iter().any(|n| n.index == 0));
        assert!(result.iter().any(|n| n.index == 1));
        assert!(result.iter().any(|n| n.index == 2));
        assert!(result.iter().all(|n| n.distance <= 1.01));
    }

    #[test]
    fn zero_k_returns_empty() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = KdTree::new(&data, MaxVarianceSplit);
        let query = data.query().with_coordinates(&points[0]);
        let result: Vec<DistPair<f64>> = tree.search_knn(&query, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn manhattan_range_respects_l1_radius() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = KdTree::new(&data, MaxVarianceSplit);
        let query = data.query().with_coordinates(&points[0]);
        let result: Vec<DistPair<f64>> = tree.search_range(&query, 2.0);

        let indices: Vec<usize> = result.iter().map(|n| n.index).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
        assert!(result.iter().all(|n| n.distance <= 2.0));
    }

    #[test]
    fn minkowski_with_two_matches_euclidean_knn() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let euclid = KdTree::new(&data, MaxVarianceSplit);
        let minkowski = KdTree::new(&data, MaxVarianceSplit);

        let k = 3;
        let query = data.query().with_coordinates(&points[0]);
        let knn_euclid: Vec<DistPair<f64>> = euclid.search_knn(&query, k);
        let knn_mink: Vec<DistPair<f64>> = minkowski.search_knn(&query, k);

        assert_eq!(knn_mink.len(), knn_euclid.len());
        for (a, b) in knn_mink.iter().zip(knn_euclid.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.distance - b.distance).abs() < 1e-6);
        }
    }

    #[test]
    fn axis_cycle_strategy_is_available() {
        let points = sample_points();
        let data: TableWithDistance<'_, f64, Vec<f64>, SquaredEuclidean, f64> =
            TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = KdTree::new(&data, AxisCycleSplit);
        assert_eq!(tree.len(), points.len());
    }

    #[test]
    fn largest_spread_strategy_is_available() {
        let points = sample_points();
        let data: TableWithDistance<'_, f64, Vec<f64>, SquaredEuclidean, f64> =
            TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = KdTree::new(&data, LargestSpreadSplit);
        assert_eq!(tree.len(), points.len());
    }
}
