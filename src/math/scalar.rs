//! Scalar (un-optimised) implementations of the vector math primitives.
//!
//! Used as a fallback when AVX2 is not available or when the element type
//! does not have a dedicated intrinsic path.
use ndarray::{ArrayView1, ArrayView2};

use crate::Float;

#[inline(always)]
pub fn sqdist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    (0..d).map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) }).map(|x| x * x).sum()
}

#[inline(always)]
pub fn sqdist_view<N>(v1: ArrayView1<'_, N>, v2: ArrayView1<'_, N>, d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let mut sum = N::zero();
    for i in 0..d {
        let diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    sum
}

#[inline(always)]
pub fn l1dist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let mut sum = N::zero();
    for i in 0..d {
        let diff = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum += diff.abs();
    }
    sum
}

#[inline(always)]
pub fn mul<N>(v1: &mut [N], v2: &[N], a: N, d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    for i in 0..d {
        unsafe {
            *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
        }
    }
}

#[inline(always)]
pub fn mul_assign<N>(v: &mut [N], f: N, d: usize)
where
    N: Float,
{
    assert!(v.len() >= d);
    for i in 0..d {
        unsafe {
            *v.get_unchecked_mut(i) *= f;
        }
    }
}

#[inline(always)]
pub fn add_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    for i in 0..d {
        unsafe {
            *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub fn sub_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    for i in 0..d {
        unsafe {
            *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub fn fmamul<N>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);

    #[cfg(any(target_feature = "fma", target_feature = "neon", target_feature = "vfp4"))]
    {
        for i in 0..d {
            unsafe {
                let fma = num_traits::Float::mul_add(*v1.get_unchecked(i), a, *v2.get_unchecked(i));
                *v1.get_unchecked_mut(i) = fma * b;
            }
        }
    }

    #[cfg(not(any(target_feature = "fma", target_feature = "neon", target_feature = "vfp4")))]
    for i in 0..d {
        unsafe {
            *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i) * a + *v2.get_unchecked(i)) * b;
        }
    }
}

#[inline(always)]
pub fn dot<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    (0..d).map(|i| unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) }).sum()
}

#[inline(always)]
pub fn pairwise_sqdist_between<N>(
    points1: ArrayView2<'_, N>, points2: ArrayView2<'_, N>, d: usize, out: &mut [N], nrows: usize,
    ncols: usize,
) where
    N: Float,
{
    assert_eq!(out.len(), nrows * ncols);
    let points1_contig = points1.is_standard_layout();
    let points2_contig = points2.is_standard_layout();

    if points1_contig && points2_contig {
        for i in 0..nrows {
            let row1_view = points1.row(i);
            let row1 = row1_view.as_slice().unwrap();
            for j in 0..ncols {
                let row2_view = points2.row(j);
                let row2 = row2_view.as_slice().unwrap();
                out[i * ncols + j] = sqdist(row1, row2, d);
            }
        }
    } else {
        for i in 0..nrows {
            let row1 = points1.row(i);
            for j in 0..ncols {
                let row2 = points2.row(j);
                out[i * ncols + j] = sqdist_view(row1, row2, d);
            }
        }
    }
}

#[inline(always)]
pub fn axpy<N>(v1: &mut [N], a: N, v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    for i in 0..d {
        unsafe {
            *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) * a;
        }
    }
}

#[inline(always)]
pub fn sum<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v.len() >= d);
    (0..d).map(|i| unsafe { *v.get_unchecked(i) }).sum()
}

#[inline(always)]
pub fn norm<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    dot(v, v, d).sqrt()
}

#[inline(always)]
pub fn add_scalar<N>(v: &mut [N], s: N, d: usize)
where
    N: Float,
{
    assert!(v.len() >= d);
    for i in 0..d {
        unsafe {
            *v.get_unchecked_mut(i) += s;
        }
    }
}

/// Squared distances from a single `center` to each of the `n` rows in `points`.
pub fn rowdist<N>(center: &[N], points: ArrayView2<'_, N>, d: usize, out: &mut [N], n: usize)
where
    N: Float,
{
    assert_eq!(out.len(), n);
    for (j, out_val) in out.iter_mut().enumerate().take(n) {
        let row = points.row(j);
        *out_val = if let Some(s) = row.as_slice() {
            sqdist(center, s, d)
        } else {
            sqdist_view(ArrayView1::from(center), row, d)
        };
    }
}
