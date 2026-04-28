//! Core mathematical primitives.
//!
//! Public API: plain free functions (`sqdist`, `dot`, `axpy`, ...).
//! Back-end selection (AVX2, unrolled scalar, plain scalar) happens at
//! monomorphisation time through the [`crate::VecOps`] trait.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(not(target_arch = "x86_64"))]
pub mod unroll;

use crate::{Float, VecOps};

/// Minimum number of dimensions to invoke the AVX2 or unrolled back-ends.
const UNROLL_SIZE: usize = 4;

// ---------------------------------------------------------------------------
// VecOps impls - back-end selected at monomorphisation, zero runtime overhead
// ---------------------------------------------------------------------------

impl VecOps for f32 {
    #[inline(always)]
    fn vec_sqdist(v1: &[f32], v2: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sqdist_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sqdist(v1, v2, d);
        }
        scalar::sqdist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_pairwise_sqdist<D1: AsRef<[f32]>, D2: AsRef<[f32]>>(
        points1: &[D1], points2: &[D2], d: usize, out: &mut [f32], nrows: usize, ncols: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::pairwise_sqdist_between_f32(points1, points2, d, out, nrows, ncols);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
        }
        scalar::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
    }
    #[inline(always)]
    fn vec_l1dist(v1: &[f32], v2: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::l1dist_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::l1dist(v1, v2, d);
        }
        scalar::l1dist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_mul(v1: &mut [f32], v2: &[f32], a: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_f32(v1, v2, a, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul(v1, v2, a, d);
        }
        scalar::mul(v1, v2, a, d)
    }
    #[inline(always)]
    fn vec_mul_assign(v: &mut [f32], f: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_assign_f32(v, f, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul_assign(v, f, d);
        }
        scalar::mul_assign(v, f, d)
    }
    #[inline(always)]
    fn vec_add_assign(v1: &mut [f32], v2: &[f32], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_assign_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_assign(v1, v2, d);
        }
        scalar::add_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_sub_assign(v1: &mut [f32], v2: &[f32], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sub_assign_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sub_assign(v1, v2, d);
        }
        scalar::sub_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_fmamul(v1: &mut [f32], a: f32, v2: &[f32], b: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::fmamul_f32(v1, a, v2, b, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::fmamul(v1, a, v2, b, d);
        }
        scalar::fmamul(v1, a, v2, b, d)
    }
    #[inline(always)]
    fn vec_dot(v1: &[f32], v2: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::dot_f32(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::dot(v1, v2, d);
        }
        scalar::dot(v1, v2, d)
    }
    #[inline(always)]
    fn vec_axpy(v1: &mut [f32], a: f32, v2: &[f32], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::axpy_f32(v1, a, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::axpy(v1, a, v2, d);
        }
        scalar::axpy(v1, a, v2, d)
    }
    #[inline(always)]
    fn vec_sum(v: &[f32], d: usize) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sum_f32(v, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sum(v, d);
        }
        scalar::sum(v, d)
    }
    #[inline(always)]
    fn vec_add_scalar(v: &mut [f32], s: f32, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_scalar_f32(v, s, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_scalar(v, s, d);
        }
        scalar::add_scalar(v, s, d)
    }
}

impl VecOps for f64 {
    #[inline(always)]
    fn vec_sqdist(v1: &[f64], v2: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sqdist_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sqdist(v1, v2, d);
        }
        scalar::sqdist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_pairwise_sqdist<D1: AsRef<[f64]>, D2: AsRef<[f64]>>(
        points1: &[D1], points2: &[D2], d: usize, out: &mut [f64], nrows: usize, ncols: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::pairwise_sqdist_between_f64(points1, points2, d, out, nrows, ncols);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
        }
        scalar::pairwise_sqdist_between(points1, points2, d, out, nrows, ncols);
    }
    #[inline(always)]
    fn vec_l1dist(v1: &[f64], v2: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::l1dist_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::l1dist(v1, v2, d);
        }
        scalar::l1dist(v1, v2, d)
    }
    #[inline(always)]
    fn vec_mul(v1: &mut [f64], v2: &[f64], a: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_f64(v1, v2, a, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul(v1, v2, a, d);
        }
        scalar::mul(v1, v2, a, d)
    }
    #[inline(always)]
    fn vec_mul_assign(v: &mut [f64], f: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::mul_assign_f64(v, f, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::mul_assign(v, f, d);
        }
        scalar::mul_assign(v, f, d)
    }
    #[inline(always)]
    fn vec_add_assign(v1: &mut [f64], v2: &[f64], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_assign_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_assign(v1, v2, d);
        }
        scalar::add_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_sub_assign(v1: &mut [f64], v2: &[f64], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sub_assign_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sub_assign(v1, v2, d);
        }
        scalar::sub_assign(v1, v2, d)
    }
    #[inline(always)]
    fn vec_fmamul(v1: &mut [f64], a: f64, v2: &[f64], b: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::fmamul_f64(v1, a, v2, b, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::fmamul(v1, a, v2, b, d);
        }
        scalar::fmamul(v1, a, v2, b, d)
    }
    #[inline(always)]
    fn vec_dot(v1: &[f64], v2: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::dot_f64(v1, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::dot(v1, v2, d);
        }
        scalar::dot(v1, v2, d)
    }
    #[inline(always)]
    fn vec_axpy(v1: &mut [f64], a: f64, v2: &[f64], d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::axpy_f64(v1, a, v2, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::axpy(v1, a, v2, d);
        }
        scalar::axpy(v1, a, v2, d)
    }
    #[inline(always)]
    fn vec_sum(v: &[f64], d: usize) -> f64 {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::sum_f64(v, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::sum(v, d);
        }
        scalar::sum(v, d)
    }
    #[inline(always)]
    fn vec_add_scalar(v: &mut [f64], s: f64, d: usize) {
        #[cfg(target_arch = "x86_64")]
        if d >= UNROLL_SIZE {
            return avx2::add_scalar_f64(v, s, d);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if d >= UNROLL_SIZE {
            return unroll::add_scalar(v, s, d);
        }
        scalar::add_scalar(v, s, d)
    }
}

// ---------------------------------------------------------------------------
// Public free functions - thin wrappers delegating to VecOps
// ---------------------------------------------------------------------------

/// Squared Euclidean distance between two length-`d` vectors.
#[inline(always)]
pub fn sqdist<N: Float>(v1: &[N], v2: &[N], d: usize) -> N { N::vec_sqdist(v1, v2, d) }

/// Pairwise squared distances between two point sets.
///
/// Returns an `nrows*ncols` row-major distance matrix for `points1 x points2`.
#[inline]
pub fn pairwise_sqdist<N: Float, D1: AsRef<[N]>, D2: AsRef<[N]>>(
    points1: &[D1], points2: &[D2], d: usize,
) -> Vec<N> {
    let nrows = points1.len();
    let ncols = points2.len();
    let mut matrix = vec![N::zero(); nrows.checked_mul(ncols).expect("point count overflow")];
    if nrows == 0 || ncols == 0 {
        return matrix;
    }
    N::vec_pairwise_sqdist(points1, points2, d, &mut matrix, nrows, ncols);
    matrix
}

/// L1 (Manhattan) distance between two vectors.
#[inline(always)]
pub fn l1dist<N: Float>(v1: &[N], v2: &[N], d: usize) -> N { N::vec_l1dist(v1, v2, d) }

/// Set `v1[i] = v2[i] * a` for `i` in `0..d`.
#[inline(always)]
pub fn mul<N: Float>(v1: &mut [N], v2: &[N], a: N, d: usize) { N::vec_mul(v1, v2, a, d) }

/// In-place multiply by a scalar: `v[i] *= f`.
#[inline(always)]
pub fn mul_assign<N: Float>(v: &mut [N], f: N, d: usize) { N::vec_mul_assign(v, f, d) }

/// Alias for `mul_assign`.
#[inline(always)]
pub fn scale<N: Float>(v: &mut [N], f: N, d: usize) { N::vec_mul_assign(v, f, d) }

/// In-place addition: `v1[i] += v2[i]`.
#[inline(always)]
pub fn add_assign<N: Float>(v1: &mut [N], v2: &[N], d: usize) { N::vec_add_assign(v1, v2, d) }

/// In-place subtraction: `v1[i] -= v2[i]`.
#[inline(always)]
pub fn sub_assign<N: Float>(v1: &mut [N], v2: &[N], d: usize) { N::vec_sub_assign(v1, v2, d) }

/// FMA followed by a multiplication: `v1 = (v1 * a + v2) * b`.
#[inline(always)]
pub fn fmamul<N: Float>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) {
    N::vec_fmamul(v1, a, v2, b, d)
}

/// Dot product of two vectors.
#[inline(always)]
pub fn dot<N: Float>(v1: &[N], v2: &[N], d: usize) -> N { N::vec_dot(v1, v2, d) }

/// Squared L2 norm of a vector.
#[inline(always)]
pub fn sqnorm<N: Float>(v: &[N], d: usize) -> N { N::vec_dot(v, v, d) }

/// Euclidean (L2) norm of a vector.
#[inline(always)]
pub fn norm<N: Float>(v: &[N], d: usize) -> N { sqnorm(v, d).sqrt() }

/// In-place scaled addition (AXPY): `v1[i] += a * v2[i]` for `i` in `0..d`.
#[inline(always)]
pub fn axpy<N: Float>(v1: &mut [N], a: N, v2: &[N], d: usize) { N::vec_axpy(v1, a, v2, d) }

/// Combined scaled sum: `v1 := a * v1 + b * v2`.
#[inline(always)]
pub fn axpby<N: Float>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) {
    N::vec_mul_assign(v1, a, d);
    N::vec_axpy(v1, b, v2, d);
}

/// Compute the sum of the first `d` elements of a slice.
#[inline(always)]
pub fn sum<N: Float>(v: &[N], d: usize) -> N { N::vec_sum(v, d) }

/// Add a scalar to every element: `v[i] += s`.
#[inline(always)]
pub fn add_scalar<N: Float>(v: &mut [N], s: N, d: usize) { N::vec_add_scalar(v, s, d) }

/// Copy `d` elements from `v2` into `v1`.
#[inline(always)]
pub fn copy<N: Copy>(v1: &mut [N], v2: &[N], d: usize) {
    debug_assert!(v1.len() >= d && v2.len() >= d);
    unsafe {
        std::ptr::copy_nonoverlapping(v2.as_ptr(), v1.as_mut_ptr(), d);
    }
}

/// Fill a slice with a constant value.
#[inline(always)]
pub fn fill<N: Copy>(v: &mut [N], val: N, d: usize) {
    debug_assert!(v.len() >= d);
    for i in 0..d {
        unsafe {
            *v.get_unchecked_mut(i) = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squared_matrix<N: Float + std::fmt::Debug + PartialEq>(points: &[&[N]], d: usize) -> Vec<N> {
        pairwise_sqdist(points, points, d)
    }

    #[test]
    fn pairwise_sqdist_f32_row_major() {
        let a = &[&[0.0f32, 1.0f32][..], &[2.0, 3.0][..], &[1.0, 4.0][..]];
        let got = squared_matrix(a, 2);
        let expected = vec![0.0, 8.0, 10.0, 8.0, 0.0, 2.0, 10.0, 2.0, 0.0];
        assert_eq!(got, expected);
    }

    #[test]
    fn pairwise_sqdist_f64_vectorized() {
        let a =
            &[&[1.0f64, 2.0, 3.0, 4.0][..], &[4.0, 3.0, 2.0, 1.0][..], &[0.0, 0.0, 0.0, 0.0][..]];
        let got = squared_matrix(a, 4);
        let expected = vec![0.0, 20.0, 30.0, 20.0, 0.0, 30.0, 30.0, 30.0, 0.0];
        assert_eq!(got, expected);
    }
}
