// `rayon` is only required when the `parallel` feature is enabled.  The
// benchmark disables this feature to measure single-threaded performance.
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Data, DistPair, DistanceData, DistanceSearch, Float, IndexQuery, KnnSearch,
    LinearScanPrioritySearcher, PrioritySearcherFactory, RangeSearch, linear_scan_knn,
    linear_scan_range,
};

/// Helper that returns the starting offset in the flattened triangle for
/// row `i` (i.e. the number of elements in all rows before row `i`).
#[inline]
pub const fn triangle_size(i: usize) -> usize { i * (i - 1) / 2 }

/// Compute the *lower triangular* distance matrix for a data set.
///
/// The returned vector contains the distances for pairs `(i,j)` with
/// `0 <= j < i < n` in row-major order.  The length of the resulting
/// vector is `n*(n-1)/2` where `n = data.len()`.
///
/// Parallelisation is performed on the outer index using `rayon`.
pub fn lower_triangular_matrix<D, S>(data: &D) -> Vec<S>
where
    D: DistanceData<S> + Sync,
    S: Float + Send,
{
    let n = data.len();
    if n < 2 {
        return Vec::new();
    }

    #[cfg(feature = "parallel")]
    {
        (1..n).into_par_iter().flat_map_iter(|i| (0..i).map(move |j| data.distance(i, j))).collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Vec::with_capacity(n.saturating_sub(1) * n / 2);
        for i in 1..n {
            for j in 0..i {
                out.push(data.distance(i, j));
            }
        }
        out
    }
}

/// Treats a condensed (lower-triangular) distance array as a dataset where the
/// only operation is `distance(i,j)`.  This is useful when the input to an
/// algorithm is already given as a condensed distance matrix rather than raw
/// points.  The underlying vector is owned.
#[derive(Debug, Clone)]
pub struct CondensedDistanceMatrix<'a, F: Float> {
    data: &'a [F],
    n: usize,
}

impl<'a, F: Float> CondensedDistanceMatrix<'a, F> {
    pub fn new(data: &'a [F], n: usize) -> Self {
        assert_eq!(data.len(), n.saturating_sub(1) * n / 2);
        Self { data, n }
    }
}

impl<'a, F: Float> Data for CondensedDistanceMatrix<'a, F> {
    fn len(&self) -> usize { self.n }
}

impl<'a, F: Float> DistanceData<F> for CondensedDistanceMatrix<'a, F> {
    type Query<'b>
        = CondensedMatrixQuery<'b, F>
    where
        Self: 'b;

    fn distance(&self, a: usize, b: usize) -> F {
        debug_assert!(a < self.n && b < self.n);
        if a == b {
            return F::zero();
        }
        let (i, j) = if a > b { (a, b) } else { (b, a) };
        self.data[(i * (i - 1)) / 2 + j]
    }

    fn query(&self) -> Self::Query<'_> { CondensedMatrixQuery::new(self) }
}

pub struct CondensedMatrixQuery<'a, F: Float> {
    data: &'a CondensedDistanceMatrix<'a, F>,
    index: usize,
}

impl<'a, F: Float> CondensedMatrixQuery<'a, F> {
    fn new(data: &'a CondensedDistanceMatrix<'a, F>) -> Self { Self { data, index: 0 } }
}

impl<F: Float> DistanceSearch<F> for CondensedMatrixQuery<'_, F> {
    fn query_distance(&self, idx: usize) -> F { self.data.distance(self.index, idx) }
}

impl<F: Float> IndexQuery<F> for CondensedMatrixQuery<'_, F> {
    fn set_index(&mut self, idx: usize) {
        debug_assert!(idx < self.data.n);
        self.index = idx;
    }
}

/// Wraps a full square matrix stored in row-major order.  It must be symmetric
/// and generally has zero on the diagonal.  Provided mainly for benchmarking or
/// converting existing matrices into the `DataAccess` trait.
#[derive(Debug, Clone)]
pub struct SquareDistanceMatrix<'a, F: Float> {
    data: &'a [F],
    n: usize,
}

impl<'a, F: Float> SquareDistanceMatrix<'a, F> {
    pub fn new(data: &'a [F], n: usize) -> Self {
        assert_eq!(data.len(), n * n);
        Self { data, n }
    }
}

impl<'a, F: Float> Data for SquareDistanceMatrix<'a, F> {
    fn len(&self) -> usize { self.n }
}

impl<'a, F: Float> DistanceData<F> for SquareDistanceMatrix<'a, F> {
    type Query<'b>
        = SquareMatrixQuery<'b, F>
    where
        Self: 'b;

    fn distance(&self, a: usize, b: usize) -> F {
        assert!(a < self.n && b < self.n);
        self.data[a * self.n + b]
    }

    fn query(&self) -> Self::Query<'_> { SquareMatrixQuery::new(self) }
}

pub struct SquareMatrixQuery<'a, F: Float> {
    data: &'a SquareDistanceMatrix<'a, F>,
    index: usize,
}

impl<'a, F: Float> SquareMatrixQuery<'a, F> {
    fn new(data: &'a SquareDistanceMatrix<'a, F>) -> Self { Self { data, index: 0 } }
}

impl<F: Float> DistanceSearch<F> for SquareMatrixQuery<'_, F> {
    fn query_distance(&self, idx: usize) -> F { self.data.distance(self.index, idx) }
}

impl<F: Float> IndexQuery<F> for SquareMatrixQuery<'_, F> {
    fn set_index(&mut self, idx: usize) {
        debug_assert!(idx < self.data.n);
        self.index = idx;
    }
}

impl<'a, F: Float, Q: DistanceSearch<F> + ?Sized> RangeSearch<F, Q>
    for CondensedDistanceMatrix<'a, F>
{
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        linear_scan_range(self, query, radius)
    }
}

impl<'a, F: Float, Q: DistanceSearch<F> + ?Sized> RangeSearch<F, Q>
    for SquareDistanceMatrix<'a, F>
{
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        linear_scan_range(self, query, radius)
    }
}

impl<'a, F: Float, Q: DistanceSearch<F> + ?Sized> KnnSearch<F, Q>
    for CondensedDistanceMatrix<'a, F>
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> {
        linear_scan_knn(self, query, k)
    }
}

impl<'a, F: Float, Q: DistanceSearch<F> + ?Sized> KnnSearch<F, Q> for SquareDistanceMatrix<'a, F> {
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> {
        linear_scan_knn(self, query, k)
    }
}

impl<'a, F: Float, Q> PrioritySearcherFactory<F, Q> for CondensedDistanceMatrix<'a, F>
where
    Q: DistanceSearch<F> + ?Sized,
{
    type Searcher<'b>
        = LinearScanPrioritySearcher<'b, F, CondensedDistanceMatrix<'b, F>>
    where
        Self: 'b,
        Q: 'b,
        F: 'b;

    fn priority_searcher<'b>(&'b self) -> Self::Searcher<'b>
    where
        Q: 'b,
    {
        LinearScanPrioritySearcher::new(self)
    }
}

impl<'a, F: Float, Q> PrioritySearcherFactory<F, Q> for SquareDistanceMatrix<'a, F>
where
    Q: DistanceSearch<F> + ?Sized,
{
    type Searcher<'b>
        = LinearScanPrioritySearcher<'b, F, SquareDistanceMatrix<'b, F>>
    where
        Self: 'b,
        Q: 'b,
        F: 'b;

    fn priority_searcher<'b>(&'b self) -> Self::Searcher<'b>
    where
        Q: 'b,
    {
        LinearScanPrioritySearcher::new(self)
    }
}
