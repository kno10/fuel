use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};

use ndarray_linalg::{Cholesky, Scalar};

use crate::{Float, VectorData as Dataset};

/// similar helper utilities shared by multiple Gaussian model implementations
/// row-major index into a dim x dim square matrix stored as a flat slice
pub(crate) fn idx(i: usize, j: usize, dim: usize) -> usize { i * dim + j }

/// makes the matrix symmetric by copying upper to lower triangle
pub(crate) fn symmetrize<N>(matrix: &mut [N], dim: usize)
where
    N: Float + Copy,
{
    for i in 0..dim {
        for j in 0..i {
            matrix[idx(j, i, dim)] = matrix[idx(i, j, dim)];
        }
    }
}

/// compute the lower triangular Cholesky factor, or return None if decomposition fails
pub(crate) fn compute_cholesky<N>(matrix: &[N], dim: usize) -> Option<Vec<N>>
where
    N: Float + Copy + Scalar + ndarray_linalg::Lapack,
{
    let a: ndarray::Array2<N> =
        ndarray::Array2::from_shape_vec((dim, dim), matrix.to_vec()).ok()?;
    match a.cholesky(ndarray_linalg::UPLO::Lower) {
        Ok(chol) => {
            let (vec, offset) = chol.into_raw_vec_and_offset();
            assert_eq!(offset, Some(0));
            Some(vec)
        }
        Err(_) => None,
    }
}

/// half the log determinant from a lower‑triangular Cholesky factor
pub(crate) fn half_log_det<N>(chol: &[N], dim: usize) -> N
where
    N: Float + Copy,
{
    let mut acc = N::zero();
    for i in 0..dim {
        acc = acc + chol[idx(i, i, dim)].ln();
    }
    acc
}

/// small jitter proportional to diagonal sum to help make matrix positive definite
pub(crate) fn jitter_amount<N>(matrix: &[N], dim: usize) -> N
where
    N: Float + Copy,
{
    let mut sum = N::zero();
    for i in 0..dim {
        sum = sum + matrix[idx(i, i, dim)];
    }
    let eps = N::from(1e-100).unwrap();
    if sum > eps {
        sum * N::from(1e-10).unwrap() / N::from(dim).unwrap()
    } else {
        N::from(1e-5).unwrap()
    }
}

/// solve L x = delta where L is lower-triangular stored in row-major form
pub(crate) fn solve_lower<N>(chol: &[N], dim: usize, delta: &[N]) -> Vec<N>
where
    N: Float + Copy,
{
    let mut solution = vec![N::zero(); dim];
    for i in 0..dim {
        let mut sum = delta[i];
        for j in 0..i {
            sum = sum - chol[idx(i, j, dim)] * solution[j];
        }
        solution[i] = sum / chol[idx(i, i, dim)];
    }
    solution
}

pub(crate) fn enforce_min_diagonal<N>(matrix: &mut [N], dim: usize, min_variance: N)
where
    N: Float + Copy,
{
    for i in 0..dim {
        let diagonal = idx(i, i, dim);
        if matrix[diagonal] < min_variance {
            matrix[diagonal] = min_variance;
        }
    }
}

pub(crate) fn refresh_cholesky_log_norm_det<N>(
    covariance: &mut [N], dim: usize, min_variance: N, weight: N, log_norm: N, chol: &mut Vec<N>,
) -> N
where
    N: Float + Copy + Scalar + ndarray_linalg::Lapack,
{
    symmetrize(covariance, dim);
    enforce_min_diagonal(covariance, dim, min_variance);

    if let Some(factor) = compute_cholesky(covariance, dim) {
        *chol = factor;
    } else {
        let jitter = jitter_amount(covariance, dim);
        for i in 0..dim {
            let diagonal = idx(i, i, dim);
            covariance[diagonal] += jitter;
        }
        if let Some(factor) = compute_cholesky(covariance, dim) {
            *chol = factor;
        }
    }

    if chol.iter().all(|&v| v == N::zero()) {
        let mut fallback = vec![N::zero(); dim * dim];
        for i in 0..dim {
            fallback[idx(i, i, dim)] = N::one();
        }
        *chol = fallback;
    }

    let half = half_log_det(chol, dim);
    num_traits::Float::ln(weight) - N::from(0.5).unwrap() * log_norm - half
}

pub(crate) fn mahalanobis_distance_from_cholesky<N>(chol: &[N], mean: &[N], x: &[N]) -> N
where
    N: Float + Copy,
{
    let dim = mean.len();
    let mut delta = vec![N::zero(); dim];
    for i in 0..dim {
        delta[i] = x[i] - mean[i];
    }
    let solution = solve_lower(chol, dim, &delta);
    solution.iter().copied().fold(N::zero(), |acc, v| acc + v * v)
}

pub(crate) fn log_norm_det_diagonal<N>(weight: N, variance: &[N], min_variance: N) -> N
where
    N: Float + Copy,
{
    let d = N::from(variance.len()).unwrap();
    let log_2pi = N::from(2.0 * std::f64::consts::PI).unwrap().ln();
    let mut log_det = N::zero();
    for &v in variance {
        log_det = log_det + v.max(min_variance).ln();
    }
    weight.ln() - N::from(0.5).unwrap() * (d * log_2pi + log_det)
}

pub(crate) fn log_norm_det_spherical<N>(weight: N, dim: usize, variance: N, min_variance: N) -> N
where
    N: Float + Copy,
{
    let d = N::from(dim).unwrap();
    let log_2pi = N::from(2.0 * std::f64::consts::PI).unwrap().ln();
    let log_det = d * variance.max(min_variance).ln();
    weight.ln() - N::from(0.5).unwrap() * (d * log_2pi + log_det)
}

use crate::math;

pub(crate) fn global_mean<N, A>(data: &A) -> Vec<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let nf = N::from(n).unwrap();
    let mut mean = vec![N::zero(); d];
    let mut scratch = vec![N::zero(); d];

    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        // use math helper to accumulate the row into the mean
        math::add_assign(&mut mean, &scratch, d);
    }

    for m in &mut mean {
        *m = *m / nf;
    }

    mean
}

pub(crate) fn scale_component_covariance<N>(
    covariance: &mut [N], k: usize, dim: usize, min_variance: N,
) where
    N: Float + Copy,
{
    let scale = N::from(k).unwrap().powf(-N::from(2.0).unwrap() / N::from(dim).unwrap());
    for value in covariance {
        *value = (*value * scale).max(min_variance);
    }
}

pub(crate) fn scale_component_variance<N>(variance: N, k: usize, dim: usize, min_variance: N) -> N
where
    N: Float + Copy,
{
    let scale = N::from(k).unwrap().powf(-N::from(2.0).unwrap() / N::from(dim).unwrap());
    (variance * scale).max(min_variance)
}
