use crate::api::search::{LinearScanPrioritySearcher, linear_scan_knn, linear_scan_range};
use crate::{
    DistPair, DistanceData, DistanceSearch, Float, KnnSearch, PrioritySearcherFactory, RangeSearch,
};

/// Linear scan searcher that operates over any `DistanceData`.
pub struct LinearScanSearcher<'a, D>
where
    D: ?Sized,
{
    data: &'a D,
}

impl<'a, D> LinearScanSearcher<'a, D>
where
    D: ?Sized,
{
    /// Create a new linear scan searcher over the given data.
    pub fn new(data: &'a D) -> Self { Self { data } }

    /// Borrow the underlying data set.
    pub fn data(&self) -> &'a D { self.data }
}

impl<'a, F, D, Q> KnnSearch<F, Q> for LinearScanSearcher<'a, D>
where
    F: Float,
    D: DistanceData<F>,
    Q: DistanceSearch<F> + ?Sized,
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> {
        linear_scan_knn(self.data, query, k)
    }
}

impl<'a, F, D, Q> RangeSearch<F, Q> for LinearScanSearcher<'a, D>
where
    F: Float,
    D: DistanceData<F>,
    Q: DistanceSearch<F> + ?Sized,
{
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        linear_scan_range(self.data, query, radius)
    }
}

impl<'a, F, D, Q> PrioritySearcherFactory<F, Q> for LinearScanSearcher<'a, D>
where
    F: Float,
    D: DistanceData<F>,
    Q: DistanceSearch<F> + ?Sized,
{
    type Searcher<'b>
        = LinearScanPrioritySearcher<'b, F, D>
    where
        Self: 'b,
        Q: 'b,
        F: 'b;

    fn priority_searcher<'b>(&'b self) -> Self::Searcher<'b>
    where
        Q: 'b,
    {
        LinearScanPrioritySearcher::new(self.data)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::LinearScanSearcher;
    use crate::api::data::DistanceData;
    use crate::api::ndarray::NdArrayDatasetWithDistance;
    use crate::distance::Euclidean;
    use crate::{CoordinateQuery, KnnSearch, RangeSearch};

    #[test]
    fn linear_scan_searcher_knn_matches_expected_order() {
        let data = array![[0.0f32, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let dataset = NdArrayDatasetWithDistance::with_distance(&data, Euclidean);
        let searcher = LinearScanSearcher::new(&dataset);

        let query = dataset.query().with_coordinates(&[1.0f32, 0.0]);

        let neighbors = searcher.search_knn(&query, 2);
        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].index, 1);
        assert_eq!(neighbors[1].index, 0);
        assert_eq!(neighbors[2].index, 2);
        assert!(neighbors[0].distance <= neighbors[1].distance);
        assert!(neighbors[1].distance <= neighbors[2].distance);
    }

    #[test]
    fn linear_scan_searcher_range_returns_points_within_radius() {
        let data = array![[0.0f64, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let dataset = NdArrayDatasetWithDistance::with_distance(&data, Euclidean);
        let searcher = LinearScanSearcher::new(&dataset);

        let query = dataset.query().with_coordinates(&[0.5f64, 0.0]);

        let neighbors = searcher.search_range(&query, 1.0);
        let indices: Vec<usize> = neighbors.into_iter().map(|pair| pair.index).collect();
        assert_eq!(indices, vec![0, 1]);
    }
}
