//! Core mathematical primitives.
//!
//! Public API: plain free functions (`sqdist`, `dot`, `axpy`, ...).
//! On x86-64 the hot paths delegate to the AVX2 back-end, which uses explicit
//! intrinsics and FMA for f32 and f64.  On other architectures the scalar
//! module is used directly.

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

use crate::Float;

/// Squared Euclidean distance between two length-`d` vectors.
#[inline(always)]
pub fn sqdist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::sqdist(v1, v2, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::sqdist(v1, v2, d);
}

/// L1 (Manhattan) distance between two vectors.
#[inline(always)]
pub fn l1dist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::l1dist(v1, v2, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::l1dist(v1, v2, d);
}

/// Set `v1[i] = v2[i] * a` for `i` in `0..d`.
#[inline(always)]
pub fn mul<N>(v1: &mut [N], v2: &[N], a: N, d: usize)
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::mul(v1, v2, a, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::mul(v1, v2, a, d);
}

/// In-place multiply by a scalar: `v[i] *= f`.
#[inline(always)]
pub fn mul_assign<N>(v: &mut [N], f: N, d: usize)
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::mul_assign(v, f, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::mul_assign(v, f, d);
}

/// Alias for `mul_assign`.
#[inline(always)]
pub fn scale<N>(v: &mut [N], f: N, d: usize)
where
    N: Float,
{
    mul_assign(v, f, d)
}

/// In-place addition: `v1[i] += v2[i]`.
#[inline(always)]
pub fn add_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::add_assign(v1, v2, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::add_assign(v1, v2, d);
}

/// In-place subtraction: `v1[i] -= v2[i]`.
#[inline(always)]
pub fn sub_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::sub_assign(v1, v2, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::sub_assign(v1, v2, d);
}

/// FMA followed by a multiplication: `v1 = (v1 * a + v2) * b`.
#[inline(always)]
pub fn fmamul<N>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::fmamul(v1, a, v2, b, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::fmamul(v1, a, v2, b, d);
}

/// Dot product of two vectors.
#[inline(always)]
pub fn dot<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::dot(v1, v2, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::dot(v1, v2, d);
}

/// Squared L2 norm of a vector.
#[inline(always)]
pub fn sqnorm<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    dot(v, v, d)
}

/// Euclidean (L2) norm of a vector.
#[inline(always)]
pub fn norm<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    sqnorm(v, d).sqrt()
}

/// In-place scaled addition (AXPY): `v1[i] += a * v2[i]` for `i` in `0..d`.
#[inline(always)]
pub fn axpy<N>(v1: &mut [N], a: N, v2: &[N], d: usize)
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::axpy(v1, a, v2, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::axpy(v1, a, v2, d);
}

/// Combined scaled sum: `v1 := a * v1 + b * v2`.
#[inline(always)]
pub fn axpby<N>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
where
    N: Float,
{
    mul_assign(v1, a, d);
    axpy(v1, b, v2, d);
}

/// Compute the sum of the first `d` elements of a slice.
#[inline(always)]
pub fn sum<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::sum(v, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::sum(v, d);
}

/// Add a scalar to every element: `v[i] += s`.
#[inline(always)]
pub fn add_scalar<N>(v: &mut [N], s: N, d: usize)
where
    N: Float,
{
    #[cfg(target_arch = "x86_64")]
    return avx2::add_scalar(v, s, d);
    #[cfg(not(target_arch = "x86_64"))]
    return scalar::add_scalar(v, s, d);
}

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
