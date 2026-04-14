#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Data, DistPair, DistanceData, DistanceSearch, Float, IndexQuery, KnnSearch,
    LinearScanPrioritySearcher, PrioritySearcherFactory, RangeSearch, linear_scan_knn,
    linear_scan_range,
};

/// Helper that returns the starting offset in the flattened triangle for
/// row `n` (i.e. the number of elements in all rows before row `n`).
#[inline(always)]
pub const fn triangle_size(n: usize) -> usize { (n - (n & 1)) / 2 * (n - 1 + (n & 1)) }

/// Compute the condensed lower-triangular pairwise matrix for a point set.
///
/// The result is a flattened lower triangle array of size `n*(n-1)/2` (row major,
/// pairs `(i,j)` with `0 <= j < i < n`).
///
/// This supports both distances and similarity functions (e.g. kernels).
pub fn compute_pairwise_condensed<D, F, K>(points: &[D], fun: &K) -> Vec<F>
where
    D: Sync,
    F: Float,
    K: Fn(&D, &D) -> F + Sync,
{
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }

    #[cfg(feature = "parallel")]
    {
        (1..n)
            .into_par_iter()
            .flat_map_iter(move |i| (0..i).map(move |j| fun(&points[i], &points[j])))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Vec::with_capacity(triangle_size(n));
        for i in 1..n {
            for j in 0..i {
                out.push(fun(&points[i], &points[j]));
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
pub struct CondensedDistanceMatrix<F: Float> {
    data: Vec<F>,
    n: usize,
    is_squared_distance: bool,
}

impl<F: Float> CondensedDistanceMatrix<F> {
    pub fn new_from_condensed(data: Vec<F>, n: usize, is_squared_distance: bool) -> Self {
        Self { data, n, is_squared_distance }
    }

    pub fn new_from_data<D>(data: &D) -> Self
    where
        D: DistanceData<F> + Sync,
    {
        let n = data.len();
        let indices: Vec<usize> = (0..n).collect();
        let condensed = compute_pairwise_condensed(&indices, &|i, j| data.distance(*i, *j));
        Self { data: condensed, n, is_squared_distance: data.is_squared_distance() }
    }

    pub fn as_slice(&self) -> &[F] { &self.data }

    pub fn into_vec(self) -> Vec<F> { self.data }
}

impl<F: Float> Data for CondensedDistanceMatrix<F> {
    fn len(&self) -> usize { self.n }
}

impl<F: Float> DistanceData<F> for CondensedDistanceMatrix<F> {
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

    fn is_squared_distance(&self) -> bool { self.is_squared_distance }
}

pub struct CondensedMatrixQuery<'a, F: Float> {
    data: &'a CondensedDistanceMatrix<F>,
    index: usize,
}

impl<'a, F: Float> CondensedMatrixQuery<'a, F> {
    fn new(data: &'a CondensedDistanceMatrix<F>) -> Self { Self { data, index: 0 } }
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

impl<F: Float, Q: DistanceSearch<F> + ?Sized> RangeSearch<F, Q> for CondensedDistanceMatrix<F> {
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        linear_scan_range(self, query, radius)
    }
}

impl<F: Float, Q: DistanceSearch<F> + ?Sized> KnnSearch<F, Q> for CondensedDistanceMatrix<F> {
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> {
        linear_scan_knn(self, query, k)
    }
}

impl<F: Float, Q> PrioritySearcherFactory<F, Q> for CondensedDistanceMatrix<F>
where
    Q: DistanceSearch<F> + ?Sized,
{
    type Searcher<'b>
        = LinearScanPrioritySearcher<'b, F, CondensedDistanceMatrix<F>>
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
