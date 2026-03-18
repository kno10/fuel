mod knn;
mod range;
mod priority;
pub mod split;

pub use priority::KdTreePrioritySearcher;
pub use split::{AxisCycleSplit, LargestSpreadSplit, MaxVarianceSplit, SplitStrategy};

use std::cmp::Ordering;

use num_traits::Float;

use crate::api::VectorData;
use crate::distance::PartialDistance;


/// A static KD-tree stored in heap order.
pub struct KdTree<F, M> {
    points: Vec<usize>,
    split_axes: Vec<usize>,
    split_values: Vec<F>,
    dims: usize,
    metric: M,
}

impl<F, M> KdTree<F, M>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
{
    /// Build a new tree from the given point set using the supplied split heuristic.
    pub fn new<P, S>(data: &P, strategy: S, metric: M) -> Self
    where
        P: VectorData<F> + ?Sized,
        S: SplitStrategy<F, P>,
    {
        let size = data.size();
        let dims = data.dims();
        assert!(
            size == 0 || dims > 0,
            "cannot index zero-dimensional points"
        );

        let mut tree = Self {
            points: vec![0; size],
            split_axes: vec![0; size],
            split_values: vec![F::zero(); size],
            dims,
            metric,
        };

        if size > 0 {
            let mut indices: Vec<usize> = (0..size).collect();
            tree.build_recursive(data, &mut indices, 0, size, 0, &strategy);
        }

        tree
    }

    /// Number of points stored in the tree.
    pub const fn len(&self) -> usize {
        self.points.len()
    }

    /// True if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Dimensionality of the indexed space.
    pub const fn dims(&self) -> usize {
        self.dims
    }

    fn build_recursive<P, S>(
        &mut self,
        data: &P,
        indices: &mut [usize],
        left: usize,
        right: usize,
        depth: usize,
        strategy: &S,
    ) where
        P: VectorData<F> + ?Sized,
        S: SplitStrategy<F, P>,
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
            data.point(*a)[axis]
                .partial_cmp(&data.point(*b)[axis])
                .unwrap_or(Ordering::Equal)
        });

        let median_idx = indices[node_idx];
        self.points[node_idx] = median_idx;
        self.split_axes[node_idx] = axis;
        self.split_values[node_idx] = data.point(median_idx)[axis];

        self.build_recursive(data, indices, left, node_idx, depth + 1, strategy);
        self.build_recursive(data, indices, node_idx + 1, right, depth + 1, strategy);
    }


    /// Create a priority searcher for incremental nearest neighbor enumeration.
    pub fn priority_searcher<'a, P>(
        &'a self,
        data: &'a P,
        query: &'a [F],
    ) -> KdTreePrioritySearcher<'a, F, M, M, P>
    where
        P: VectorData<F> + ?Sized,
    {
        self.priority_searcher_with_metric(data, query, self.metric.clone())
    }

    /// Create a priority searcher using a custom partial distance metric.
    pub fn priority_searcher_with_metric<'a, P, D>(
        &'a self,
        data: &'a P,
        query: &'a [F],
        metric: D,
    ) -> KdTreePrioritySearcher<'a, F, M, D, P>
    where
        P: VectorData<F> + ?Sized,
        D: PartialDistance<F> + Clone,
    {
        KdTreePrioritySearcher::new(self, data, query, metric)
    }

    pub(super) fn check_query(&self, query: &[F]) {
        assert!(query.len() >= self.dims, "query point is too short");
    }
}

impl<F, M, D> crate::KnnSearch<F, D> for KdTree<F, M>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
    D: crate::DistanceData<F> + crate::VectorData<F> + ?Sized,
{
    fn search_knn_by_index(&self, data: &D, query_idx: usize, k: usize) -> Vec<crate::DistPair<F>> {
        self.search_knn(data, data.point(query_idx), k)
    }
}

impl<F, M, D> crate::RangeSearch<F, D> for KdTree<F, M>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
    D: crate::DistanceData<F> + crate::VectorData<F> + ?Sized,
{
    fn search_range_by_index(&self, data: &D, query_idx: usize, radius: F) -> Vec<crate::DistPair<F>> {
        self.search_range(data, data.point(query_idx), radius)
    }
}

impl<'a, F, M, D, P> crate::PrioritySearcherCore<F> for KdTreePrioritySearcher<'a, F, M, D, P>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
    D: PartialDistance<F> + Clone,
    P: VectorData<F> + ?Sized,
{
    fn reset(&mut self) {
        KdTreePrioritySearcher::reset(self);
    }

    fn set_query<'b, DD: crate::DistanceData<F> + ?Sized>(&mut self, _data: &'b DD, query: &[F]) {
        let _ = _data;
        self.search(query);
    }

    fn next(&mut self) -> Option<crate::DistPair<F>> {
        Iterator::next(self)
    }

    fn all_lower_bound(&self) -> F {
        self.all_lower_bound()
    }

    fn decrease_cutoff(&mut self, threshold: F) {
        self.decrease_cutoff(threshold);
    }
}

impl<F, M, D> crate::PrioritySearch<F, D> for KdTree<F, M>
where
    F: Float + Copy,
    M: PartialDistance<F> + Clone,
    D: crate::DistanceData<F> + crate::VectorData<F> + ?Sized,
{
    type Searcher<'a> = KdTreePrioritySearcher<'a, F, M, M, D>
    where
        F: 'a,
        M: 'a,
        D: 'a;

    fn priority_searcher<'a>(&'a self, data: &'a D, query: &'a [F]) -> Self::Searcher<'a> {
        self.priority_searcher(data, query)
    }
}

#[cfg(test)]
mod tests {
    use super::{AxisCycleSplit, KdTree, LargestSpreadSplit, MaxVarianceSplit};
    use crate::TableWithDistance;
    use crate::distance::{ManhattanDistance, MinkowskiDistance, SquaredEuclideanDistance};

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
    fn search_knn_returns_nearest() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit, SquaredEuclideanDistance);
        let neighbors = tree.search_knn(&data, &points[0], 3);

        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].index, 0);
        assert!(neighbors.iter().any(|n| n.index == 1));
        assert!(neighbors.iter().any(|n| n.index == 2));
    }

    #[test]
    fn range_search_finds_close_points() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit, SquaredEuclideanDistance);
        let result = tree.search_range(&data, &points[0], 1.01);

        assert!(result.iter().any(|n| n.index == 0));
        assert!(result.iter().any(|n| n.index == 1));
        assert!(result.iter().any(|n| n.index == 2));
        assert!(result.iter().all(|n| n.distance <= 1.01));
    }

    #[test]
    fn zero_k_returns_empty() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit, SquaredEuclideanDistance);
        let result = tree.search_knn(&data, &points[0], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn manhattan_range_respects_l1_radius() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, MaxVarianceSplit, ManhattanDistance);
        let result = tree.search_range(&data, &points[0], 2.0);

        let indices: Vec<usize> = result.iter().map(|n| n.index).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
        assert!(result.iter().all(|n| n.distance <= 2.0));
    }

    #[test]
    fn minkowski_with_two_matches_euclidean_knn() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let euclid = KdTree::new(&data, MaxVarianceSplit, SquaredEuclideanDistance);
        let minkowski = KdTree::new(&data, MaxVarianceSplit, MinkowskiDistance::new(2.0));

        let k = 3;
        let knn_euclid = euclid.search_knn(&data, &points[0], k);
        let knn_mink = minkowski.search_knn(&data, &points[0], k);

        assert_eq!(knn_mink.len(), knn_euclid.len());
        for (a, b) in knn_mink.iter().zip(knn_euclid.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.distance - b.distance).abs() < 1e-6);
        }
    }

    #[test]
    fn axis_cycle_strategy_is_available() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, AxisCycleSplit, SquaredEuclideanDistance);
        assert_eq!(tree.len(), points.len());
    }

    #[test]
    fn largest_spread_strategy_is_available() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclideanDistance);
        let tree = KdTree::new(&data, LargestSpreadSplit, SquaredEuclideanDistance);
        assert_eq!(tree.len(), points.len());
    }
}
