//! Unrolled (LANES = 8) implementations of the vector math primitives.
//!
//! Used on non-x86_64 architectures (e.g. ARM) when d >= UNROLL_SIZE, giving
//! the compiler a fixed-width inner loop body suitable for auto-vectorisation
//! (e.g. ARM NEON / VFPv4).

use std::any::TypeId;

use crate::Float;

const LANES: usize = 8;

#[inline(always)]
pub(super) fn sqdist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                let x = *vv.get_unchecked(j) - *cc.get_unchecked(j);
                *vsum.get_unchecked_mut(j) += x * x;
            }
        }
    }
    let mut sum = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum += x * x;
    }
    sum
}

#[inline(always)]
pub(super) fn l1dist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                let diff = *vv.get_unchecked(j) - *cc.get_unchecked(j);
                *vsum.get_unchecked_mut(j) += diff.abs();
            }
        }
    }
    let mut sum = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        let diff = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
        sum += diff.abs();
    }
    sum
}

#[inline(always)]
pub(super) fn mul<N>(v1: &mut [N], v2: &[N], a: N, d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) = *b2.get_unchecked(j) * a;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a;
        }
    }
}

#[inline(always)]
pub(super) fn mul_assign<N>(v: &mut [N], f: N, d: usize)
where
    N: Float,
{
    assert!(v.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let b = &mut v[i..(i + LANES)];
        for j in 0..LANES {
            unsafe {
                *b.get_unchecked_mut(j) *= f;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v.get_unchecked_mut(i) *= f;
        }
    }
}

#[inline(always)]
pub(super) fn add_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) += *b2.get_unchecked(j);
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub(super) fn sub_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) -= *b2.get_unchecked(j);
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
        }
    }
}

#[inline(always)]
pub(super) fn fmamul<N>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);

    #[cfg(any(
        target_feature = "fma",
        target_feature = "neon",
        target_feature = "vfp4",
        target_feature = "vfpv4"
    ))]
    if TypeId::of::<N>() == TypeId::of::<f32>() || TypeId::of::<N>() == TypeId::of::<f64>() {
        for i in (0..sd).step_by(LANES) {
            let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
            for j in 0..LANES {
                unsafe {
                    let fma =
                        num_traits::Float::mul_add(*b1.get_unchecked(j), a, *b2.get_unchecked(j));
                    *b1.get_unchecked_mut(j) = fma * b;
                }
            }
        }
        for i in sd..d {
            unsafe {
                let fma = num_traits::Float::mul_add(*v1.get_unchecked(i), a, *v2.get_unchecked(i));
                *v1.get_unchecked_mut(i) = fma * b;
            }
        }
        return;
    }

    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) = (*b1.get_unchecked(j) * a + *b2.get_unchecked(j)) * b;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) = (*v1.get_unchecked(i) * a + *v2.get_unchecked(i)) * b;
        }
    }
}

#[inline(always)]
pub(super) fn dot<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let (vv, cc) = (&v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *vsum.get_unchecked_mut(j) += *vv.get_unchecked(j) * *cc.get_unchecked(j);
            }
        }
    }
    let mut sum = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        sum += unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) };
    }
    sum
}

#[inline(always)]
pub(super) fn axpy<N>(v1: &mut [N], a: N, v2: &[N], d: usize)
where
    N: Float,
{
    assert!(v1.len() >= d && v2.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let (b1, b2) = (&mut v1[i..(i + LANES)], &v2[i..(i + LANES)]);
        for j in 0..LANES {
            unsafe {
                *b1.get_unchecked_mut(j) += *b2.get_unchecked(j) * a;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) * a;
        }
    }
}

#[inline(always)]
pub(super) fn sum<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    assert!(v.len() >= d);
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];
    for i in (0..sd).step_by(LANES) {
        let chunk = &v[i..(i + LANES)];
        for j in 0..LANES {
            unsafe {
                *vsum.get_unchecked_mut(j) += *chunk.get_unchecked(j);
            }
        }
    }
    let mut s = vsum.iter().copied().sum::<N>();
    for i in sd..d {
        s += unsafe { *v.get_unchecked(i) };
    }
    s
}

#[inline(always)]
pub(super) fn add_scalar<N>(v: &mut [N], s: N, d: usize)
where
    N: Float,
{
    assert!(v.len() >= d);
    let sd = d & !(LANES - 1);
    for i in (0..sd).step_by(LANES) {
        let b = &mut v[i..(i + LANES)];
        for j in 0..LANES {
            unsafe {
                *b.get_unchecked_mut(j) += s;
            }
        }
    }
    for i in sd..d {
        unsafe {
            *v.get_unchecked_mut(i) += s;
        }
    }
}
