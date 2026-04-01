//! Scalar (un-optimised) implementations of the vector math primitives.
//!
//! Used as a fallback when AVX2 is not available or when the element type
//! does not have a dedicated intrinsic path.

use crate::Float;
use std::any::TypeId;

#[inline(always)]
pub fn sqdist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    (0..d).map(|i| unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) }).map(|x| x * x).sum()
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
        if TypeId::of::<N>() == TypeId::of::<f32>() || TypeId::of::<N>() == TypeId::of::<f64>() {
            for i in 0..d {
                unsafe {
                    let fma = num_traits::Float::mul_add(*v1.get_unchecked(i), a, *v2.get_unchecked(i));
                    *v1.get_unchecked_mut(i) = fma * b;
                }
            }
            return;
        }
    }

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
