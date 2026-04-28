use ndarray::Array2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Data, DistPair, DistanceData, DistanceSearch, Float, IndexQuery, KnnSearch,
    LinearScanPrioritySearcher, PrioritySearcherFactory, RangeSearch, linear_scan_knn,
    linear_scan_range,
};

/// Compute a dense pairwise matrix. Can be used with both distances and similarity functions such as kernels.
#[cfg(feature = "parallel")]
pub fn compute_pairwise_dense<D, F, K>(points: &[D], fun: &K) -> Array2<F>
where
    D: Sync,
    F: Float,
    K: Fn(&D, &D) -> F + Sync,
{
    let n = points.len();
    if n == 0 {
        return Array2::default((0, 0));
    }

    let mut matrix = Array2::from_elem((n, n), F::default());
    let matrix_slice = matrix.as_slice_mut().expect("matrix view should be contiguous");
    let matrix_ptr_addr = matrix_slice.as_mut_ptr() as usize;

    matrix_slice.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        let matrix_ptr = matrix_ptr_addr as *mut F;
        for j in i..n {
            let val = fun(&points[i], &points[j]);
            row[j] = val;
            if j > i {
                unsafe {
                    *matrix_ptr.add(j * n + i) = val;
                }
            }
        }
    });

    matrix
}

#[cfg(not(feature = "parallel"))]
pub fn compute_pairwise_dense<D, F, K>(points: &[D], fun: &K) -> Array2<F>
where
    D: Sync,
    F: Float,
    K: Fn(&D, &D) -> F + Sync,
{
    let n = points.len();
    if n == 0 {
        return Array2::default((0, 0));
    }

    let mut matrix = Array2::from_elem((n, n), F::default());
    for i in 0..n {
        for j in i..n {
            let val = fun(&points[i], &points[j]);
            matrix[[i, j]] = val;
            matrix[[j, i]] = val;
        }
    }
    matrix
}

/// Wraps a full square matrix stored in row-major order.  It must be symmetric
/// and generally has zero on the diagonal.  Provided mainly for benchmarking or
/// converting existing matrices into the `DataAccess` trait.
#[derive(Debug, Clone)]
pub struct SquareDistanceMatrix<F: Float> {
    data: Vec<F>,
    n: usize,
    is_squared_distance: bool,
}

impl<F: Float> SquareDistanceMatrix<F> {
    pub fn new_from_array2(matrix: Array2<F>, is_squared_distance: bool) -> Self {
        let shape = matrix.shape();
        assert_eq!(shape.len(), 2);
        let n = shape[0];
        assert_eq!(shape[1], n);
        let (data, offset) = matrix.into_raw_vec_and_offset();
        assert_eq!(offset, Some(0));
        Self { data, n, is_squared_distance }
    }

    pub fn new_from_data<D>(data: &D) -> Self
    where
        D: DistanceData<F> + Sync,
    {
        let n = data.len();
        let points: Vec<usize> = (0..n).collect();
        Self::new_from_array2(
            compute_pairwise_dense(&points, &|i, j| data.distance(*i, *j)),
            data.is_squared_distance(),
        )
    }

    pub fn as_slice(&self) -> &[F] { &self.data }
}

impl<F: Float> Data for SquareDistanceMatrix<F> {
    fn len(&self) -> usize { self.n }
}

impl<F: Float> DistanceData<F> for SquareDistanceMatrix<F> {
    type Query<'b>
        = SquareMatrixQuery<'b, F>
    where
        Self: 'b;

    fn distance(&self, a: usize, b: usize) -> F {
        debug_assert!(a < self.n && b < self.n);
        self.data[a * self.n + b]
    }

    fn query(&self) -> Self::Query<'_> { SquareMatrixQuery::new(self) }

    fn is_squared_distance(&self) -> bool { self.is_squared_distance }
}

pub struct SquareMatrixQuery<'a, F: Float> {
    data: &'a SquareDistanceMatrix<F>,
    index: usize,
}

impl<'a, F: Float> SquareMatrixQuery<'a, F> {
    fn new(data: &'a SquareDistanceMatrix<F>) -> Self { Self { data, index: 0 } }
}

impl<F: Float> DistanceSearch<F> for SquareMatrixQuery<'_, F> {
    fn query_distance(&self, idx: usize) -> F { self.data.distance(self.index, idx) }
}

impl<F: Float> IndexQuery<F> for SquareMatrixQuery<'_, F> {
    fn set_index(&mut self, idx: usize) {
        debug_assert!(idx < self.data.n);
        self.index = idx;
    }

    fn query_index(&self) -> usize { self.index }
}

impl<F: Float, Q: DistanceSearch<F> + ?Sized> RangeSearch<F, Q> for SquareDistanceMatrix<F> {
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        linear_scan_range(self, query, radius)
    }
}

impl<F: Float, Q: DistanceSearch<F> + ?Sized> KnnSearch<F, Q> for SquareDistanceMatrix<F> {
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> {
        linear_scan_knn(self, query, k)
    }
}

impl<F: Float, Q> PrioritySearcherFactory<F, Q> for SquareDistanceMatrix<F>
where
    Q: DistanceSearch<F> + ?Sized,
{
    type Searcher<'b>
        = LinearScanPrioritySearcher<'b, F, SquareDistanceMatrix<F>>
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
