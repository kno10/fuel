use crate::api::{Data, DistanceData, DistanceSearch};
use crate::{Float, IndexQuery};

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
    fn size(&self) -> usize { self.n }
}

impl<'a, F: Float> DistanceData<F> for CondensedDistanceMatrix<'a, F> {
    type Query<'b>
        = CondensedMatrixQuery<'b, F>
    where
        Self: 'b;

    fn distance(&self, a: usize, b: usize) -> F {
        assert!(a < self.n && b < self.n);
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
    fn size(&self) -> usize { self.n }
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
