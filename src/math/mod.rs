//! Core mathematical kernel definitions used throughout the crate.
//!
//! This module defines the generic `Math` trait (formerly `KMath`) and a few
//! concrete implementations (`DefaultMath`, `UnrollMath`, `AVX2Math`).  The
//! implementations are split into submodules so they can be compared and
//! maintained independently.

mod avx2;
mod default;
mod unroll;

pub use avx2::AVX2Math;
pub use default::DefaultMath;
use num_traits::Float; //FIXME: use crate::Float instead
pub use unroll::UnrollMath;

/// Generic math operations on slices.
///
/// The trait is parameterized by the element type and provides a handful of
/// primitives such as dot products, scalar multiplication, and copy/fill
/// helpers.  Concrete implementations (in `default.rs`, `unroll.rs`,
/// `avx2.rs`) may expose specialized versions optimized for particular
/// architectures or vector widths.
pub trait Math<N> {
    /// Squared Euclidean distance between two length-`d` vectors.
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N;

    /// L1 (Manhattan) distance between two vectors.
    fn l1dist(v1: &[N], v2: &[N], d: usize) -> N;

    /// Set `v1[i] = v2[i] * a` for `i` in `0..d`.
    fn mul(v1: &mut [N], v2: &[N], a: N, d: usize);

    /// In-place multiply by a scalar: `v[i] *= f`.
    fn mul_assign(v: &mut [N], f: N, d: usize);

    /// Alias for the same operation (more descriptive when scaling).
    fn scale(v: &mut [N], f: N, d: usize) { Self::mul_assign(v, f, d); }

    /// In-place addition: `v1[i] += v2[i]`.
    fn add_assign(v1: &mut [N], v2: &[N], d: usize);

    /// In-place subtraction: `v1[i] -= v2[i]`.
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize);

    /// Convenience alias for `add_assign`.
    fn add_to(v1: &mut [N], v2: &[N], d: usize) { Self::add_assign(v1, v2, d); }

    /// Copy `d` elements from `v2` into `v1`.
    fn copy(v1: &mut [N], v2: &[N], d: usize)
    where
        N: Copy,
    {
        debug_assert!(v1.len() >= d && v2.len() >= d);
        unsafe {
            let dst = v1.as_mut_ptr();
            let src = v2.as_ptr();
            std::ptr::copy_nonoverlapping(src, dst, d);
        }
    }

    /// Fill a slice with a constant value.
    fn fill(v: &mut [N], val: N, d: usize)
    where
        N: Copy,
    {
        debug_assert!(v.len() >= d);
        unsafe {
            for i in 0..d {
                *v.get_unchecked_mut(i) = val;
            }
        }
    }

    /// FMA followed by a multiplication: `v1 = (v1 * a + v2) * b`.
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize);

    /// Dot product of two vectors.
    fn dot(v1: &[N], v2: &[N], d: usize) -> N;

    /// Squared L2 norm of a vector.
    fn sqnorm(v: &[N], d: usize) -> N { Self::dot(v, v, d) }

    /// Euclidean norm (L2) of a vector.
    fn norm(v: &[N], d: usize) -> N
    where
        N: Float,
    {
        Self::sqnorm(v, d).sqrt()
    }

    /// In-place scaled addition (AXPY): `v1[i] += a * v2[i]` for `i` in
    /// `0..d`.
    fn axpy(v1: &mut [N], a: N, v2: &[N], d: usize);

    /// Combined scaled sum: `v1 := a * v1 + b * v2`.
    fn axpby(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
    where
        N: Copy,
    {
        Self::mul_assign(v1, a, d);
        Self::axpy(v1, b, v2, d);
    }

    /// Compute the sum of the first `d` elements of a slice.
    fn sum(v: &[N], d: usize) -> N;

    /// Add a scalar to every element in the slice (`v[i] += s`).
    fn add_scalar(v: &mut [N], s: N, d: usize);
}

// Math dispatch is removed: always use DefaultMath.
// This crate now intentionally avoids runtime math backend selection.
