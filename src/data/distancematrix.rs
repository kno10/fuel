use crate::api::{Data, DistanceData, DistanceSearch};
use num_traits::Float;

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
    fn size(&self) -> usize {
        self.n
    }
}

impl<'a, F: Float> DistanceData<F> for CondensedDistanceMatrix<'a, F> {
    fn distance(&self, a: usize, b: usize) -> F {
        assert!(a < self.n && b < self.n);
        let (i, j) = if a > b { (a, b) } else { (b, a) };
        self.data[(i * (i - 1)) / 2 + j]
    }
    fn search_by_index(&self, idx: usize) -> impl DistanceSearch<F> {
        debug_assert!(idx < self.n);
        CondensedMatrixDistanceSearch {
            data: self,
            query: idx,
        }
    }
}

struct CondensedMatrixDistanceSearch<'a, F: Float> {
    data: &'a CondensedDistanceMatrix<'a, F>,
    query: usize,
}

impl<F: Float> DistanceSearch<F> for CondensedMatrixDistanceSearch<'_, F> {
    fn query_distance(&self, idx: usize) -> F {
        self.data.distance(self.query, idx)
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
    fn size(&self) -> usize {
        self.n
    }
}

impl<'a, F: Float> DistanceData<F> for SquareDistanceMatrix<'a, F> {
    fn distance(&self, a: usize, b: usize) -> F {
        assert!(a < self.n && b < self.n);
        self.data[a * self.n + b]
    }

    fn search_by_index(&self, idx: usize) -> impl DistanceSearch<F> {
        debug_assert!(idx < self.n);
        SquareMatrixDistanceSearch {
            data: self,
            query: idx,
        }
    }
}

struct SquareMatrixDistanceSearch<'a, F: Float> {
    data: &'a SquareDistanceMatrix<'a, F>,
    query: usize,
}

impl<F: Float> DistanceSearch<F> for SquareMatrixDistanceSearch<'_, F> {
    fn query_distance(&self, idx: usize) -> F {
        self.data.distance(self.query, idx)
    }
}
