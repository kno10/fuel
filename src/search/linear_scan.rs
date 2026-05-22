use crate::api::search::{linear_scan_knn, linear_scan_range};
use crate::{
    DistPair, DistanceData, DistanceSearch, Float, KnnSearch, PrioritySearcher,
    PrioritySearcherFactory, RangeSearch, SearchFilter,
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

/// Priority searcher performing linear scan on source data.
pub struct LinearScanPrioritySearcher<'a, F, D>
where
    F: Float,
    D: DistanceData<F> + ?Sized,
{
    data: &'a D,
    candidates: Vec<DistPair<F>>,
    position: usize,
    cutoff: F,
    skip: F,
    initialized: bool,
}

impl<'a, F, D> LinearScanPrioritySearcher<'a, F, D>
where
    F: Float,
    D: DistanceData<F> + ?Sized,
{
    pub fn new(data: &'a D) -> Self {
        Self {
            data,
            candidates: Vec::new(),
            position: 0,
            cutoff: F::infinity(),
            skip: F::zero(),
            initialized: false,
        }
    }

    fn reset_state(&mut self) {
        self.position = 0;
        self.cutoff = F::infinity();
        self.skip = F::zero();
        self.initialized = false;
    }

    fn ensure_initialized<QI>(&mut self, query: &QI)
    where
        QI: DistanceSearch<F> + ?Sized,
    {
        if self.initialized {
            return;
        }

        let n = self.data.len();
        let mut candidates = Vec::with_capacity(n);

        for i in 0..n {
            let d = query.query_distance(i);
            if d < self.skip || d > self.cutoff {
                continue;
            }
            candidates.push(DistPair::new(d, i));
        }

        candidates.sort();

        self.candidates = candidates;
        self.position = 0;
        self.initialized = true;
    }

    fn find_next<QI, S>(&mut self, query: &QI, filter: &mut S) -> Option<DistPair<F>>
    where
        QI: DistanceSearch<F> + ?Sized,
        S: SearchFilter + ?Sized,
    {
        self.ensure_initialized(query);

        while let Some(candidate) = self.candidates.get(self.position).copied() {
            self.position += 1;

            if candidate.distance > self.cutoff {
                return None;
            }
            if candidate.distance < self.skip {
                continue;
            }
            if filter.skip_point(candidate.index) {
                continue;
            }
            return Some(candidate);
        }

        None
    }

    fn all_lower_bound_internal(&self) -> F {
        if !self.initialized {
            return F::zero();
        }

        for i in self.position..self.candidates.len() {
            let d = self.candidates[i].distance;
            if d > self.cutoff {
                break;
            }
            if d >= self.skip {
                return d;
            }
        }

        F::infinity()
    }

    fn decrease_cutoff_internal(&mut self, threshold: F) {
        debug_assert!(threshold <= self.cutoff, "Thresholds must only decrease.");
        self.cutoff = threshold;
    }

    pub fn reset(&mut self) { self.reset_state(); }

    pub fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        debug_assert!(skip >= F::zero(), "Skip threshold must be non-negative.");
        debug_assert!(cutoff >= skip, "Cutoff must be >= skip threshold.");
        self.reset_state();
        self.cutoff = cutoff;
        self.skip = skip;
    }

    pub fn all_lower_bound(&self) -> F { self.all_lower_bound_internal() }

    pub fn decrease_cutoff(&mut self, threshold: F) { self.decrease_cutoff_internal(threshold); }
}

struct PointFilter<P> {
    skip_point: P,
}

impl<P> SearchFilter for PointFilter<P>
where
    P: FnMut(usize) -> bool,
{
    fn skip_point(&mut self, index: usize) -> bool { (self.skip_point)(index) }
}

impl<F, Q, D> PrioritySearcher<F, Q> for LinearScanPrioritySearcher<'_, F, D>
where
    F: Float,
    Q: DistanceSearch<F> + ?Sized,
    D: DistanceData<F> + ?Sized,
{
    fn reset(&mut self) { self.reset_state(); }

    fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        debug_assert!(skip >= F::zero(), "Skip threshold must be non-negative.");
        debug_assert!(cutoff >= skip, "Cutoff must be >= skip threshold.");
        self.reset_state();
        self.cutoff = cutoff;
        self.skip = skip;
    }

    fn next(&mut self, query: &Q) -> Option<DistPair<F>> {
        self.next_with_filter(query, &mut PointFilter { skip_point: |_| false })
    }

    fn next_with_filter(
        &mut self, query: &Q, filter: &mut dyn SearchFilter,
    ) -> Option<DistPair<F>> {
        self.find_next(query, filter)
    }

    fn all_lower_bound(&self) -> F { self.all_lower_bound_internal() }

    fn decrease_cutoff(&mut self, threshold: F) { self.decrease_cutoff_internal(threshold); }
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
