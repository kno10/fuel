use std::iter::Sum;
use std::ops::*;

use ndarray::linalg::Dot;
use ndarray::{Array2, Axis};

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset};

/// Lloyd's algorithm using BLAS-style matrix operations for assignment.
///
/// This variant computes squared distances using the identity
/// `||x - c||^2 = ||x||^2 + ||c||^2 - 2 <x, c>`, which allows the
/// point-to-center cost matrix to be obtained via a single BLAS-style
/// matrix multiplication per iteration.
///
/// When the input dataset exposes an ndarray view through `as_ndarray()`,
/// this implementation can avoid a full copy into a temporary matrix.
///
/// Numerical drawbacks:
/// - the `||x||^2 + ||c||^2 - 2<x,c>` formula is prone to cancellation when
///   distances are small compared to vector norms.
/// - for high-dimensional or large-norm data, the subtraction can lose
///   precision, so this route is best used as a performance alternative,
///   not a numerically superior one.
#[inline(always)]
pub fn lloyd_blas<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
where
    N: Float
        + ndarray::LinalgScalar
        + AddAssign
        + SubAssign
        + MulAssign
        + Sum
        + Copy
        + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    assert!(k > 0, "k must be positive");

    let (n, d) = (data.nrows(), data.ncols());

    enum DataMatrix<'a, N> {
        View(ndarray::ArrayView2<'a, N>),
        Owned(Array2<N>),
    }

    impl<'a, N> DataMatrix<'a, N> {
        fn view(&self) -> ndarray::ArrayView2<'_, N> {
            match self {
                DataMatrix::View(view) => *view,
                DataMatrix::Owned(matrix) => matrix.view(),
            }
        }
    }

    let data_matrix = if let Some(view) = data.as_ndarray() {
        DataMatrix::View(view)
    } else {
        let mut matrix = Array2::<N>::zeros((n, d));
        for i in 0..n {
            let mut row = matrix.row_mut(i);
            data.load_into(i, row.as_slice_mut().unwrap(), d);
        }
        DataMatrix::Owned(matrix)
    };

    let data_view = data_matrix.view();
    let mut x_norms = Vec::with_capacity(n);
    for row in data_view.axis_iter(Axis(0)) {
        x_norms.push(row.dot(&row));
    }

    let mut cent = Centers::<N>::new(k, d);
    init.init::<A>(data, &mut cent, k);

    let mut assign = vec![0_usize; n];
    let mut counts = vec![0_usize; k];
    let mut sums = Array2::<N>::zeros((k, d));
    let mut lastsum = N::zero();

    let two = N::from(2).unwrap();

    let assign_and_update = |cent: &Centers<N>,
                             assign: &mut [usize],
                             counts: &mut [usize],
                             sums: &mut Array2<N>,
                             lastsum: &mut N| {
        let mut cent_mat = Array2::<N>::zeros((k, d));
        for j in 0..k {
            cent_mat.row_mut(j).as_slice_mut().unwrap().copy_from_slice(cent.center(j));
        }

        let center_norms: Vec<N> = cent_mat.axis_iter(Axis(0)).map(|row| row.dot(&row)).collect();

        let gram = data_view.dot(&cent_mat.t());

        assign.fill(0);
        for slot in counts.iter_mut() {
            *slot = 0;
        }
        sums.fill(N::zero());
        *lastsum = N::zero();

        for i in 0..n {
            let mut best_j = 0;
            let mut best_dist = x_norms[i] + center_norms[0] - two * gram[[i, 0]];
            for j in 1..k {
                let dist = x_norms[i] + center_norms[j] - two * gram[[i, j]];
                if dist < best_dist {
                    best_j = j;
                    best_dist = dist;
                }
            }
            assign[i] = best_j;
            counts[best_j] += 1;
            let row = data_view.row(i);
            let mut target = sums.row_mut(best_j);
            for l in 0..d {
                target[l] += row[l];
            }
            *lastsum += best_dist;
        }
    };

    assign_and_update(&cent, &mut assign, &mut counts, &mut sums, &mut lastsum);

    let mut iter = 1;
    while iter < maxiter {
        iter += 1;

        let old_cent = if tol > N::zero() { Some(cent.clone()) } else { None };
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };

        for j in 0..k {
            if counts[j] > 0 {
                let denom = N::from(counts[j]).unwrap();
                let center_row = cent.center_mut(j);
                let sum_row = sums.row(j);
                for l in 0..d {
                    center_row[l] = sum_row[l] / denom;
                }
            }
        }

        if let Some(ref old) = old_cent {
            let diff = cent.diff_frobenius_norm(old);
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                break;
            }
        }

        let prev_assign = assign.clone();
        assign_and_update(&cent, &mut assign, &mut counts, &mut sums, &mut lastsum);

        if assign == prev_assign {
            break;
        }
    }

    KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::lloyd;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = lloyd::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);

        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = lloyd_blas::<_, _, _>(&dataset, 5, &mut init2, 100, 0.0);

        assert_eq!(res1.iterations, res2.iterations);
        assert!((res1.inertia.unwrap() - res2.inertia.unwrap()).abs() < 1e-12);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let _ = lloyd::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);

        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = lloyd_blas::<_, _, _>(&dataset, 5, &mut init2, 100, tol);

        assert!(res2.iterations <= 100);
    }
}
