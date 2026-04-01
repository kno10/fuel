//! AVX2-accelerated implementations of the vector math primitives.
//!
//! Only compiled on x86-64 (declared behind `#[cfg(target_arch = "x86_64")]` in
//! the parent module).  Each function dispatches on element type via `TypeId`:
//! f32 and f64 get explicit AVX2 intrinsics; other types fall back to scalar.
//! Results for f32/f64 may differ slightly from scalar due to FMA rounding.

use std::any::TypeId;
use std::arch::x86_64::*;

use crate::Float;

#[inline(always)]
pub(super) fn sqdist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&[N], &[f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_ps() };
        while i + 8 <= d {
            unsafe {
                let a = _mm256_loadu_ps(v1.as_ptr().add(i));
                let b = _mm256_loadu_ps(v2.as_ptr().add(i));
                let diff = _mm256_sub_ps(a, b);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }
            i += 8;
        }
        let mut sum: f32 = 0.0;
        let mut buf = [0f32; 8];
        unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
        sum += buf.iter().copied().sum::<f32>();
        while i < d {
            let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
            sum += x * x;
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&[N], &[f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_pd() };
        while i + 4 <= d {
            unsafe {
                let a = _mm256_loadu_pd(v1.as_ptr().add(i));
                let b = _mm256_loadu_pd(v2.as_ptr().add(i));
                let diff = _mm256_sub_pd(a, b);
                acc = _mm256_fmadd_pd(diff, diff, acc);
            }
            i += 4;
        }
        let mut sum: f64 = 0.0;
        let mut buf = [0f64; 4];
        unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
        sum += buf.iter().copied().sum::<f64>();
        while i < d {
            let x = unsafe { *v1.get_unchecked(i) - *v2.get_unchecked(i) };
            sum += x * x;
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    super::scalar::sqdist(v1, v2, d)
}

#[inline(always)]
pub(super) fn l1dist<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&[N], &[f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_ps() };
        let abs_mask = unsafe { _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff)) };
        while i + 8 <= d {
            unsafe {
                let a = _mm256_loadu_ps(v1.as_ptr().add(i));
                let b = _mm256_loadu_ps(v2.as_ptr().add(i));
                let diff = _mm256_sub_ps(a, b);
                let abs = _mm256_and_ps(diff, abs_mask);
                acc = _mm256_add_ps(acc, abs);
            }
            i += 8;
        }
        let mut sum: f32 = 0.0;
        let mut buf = [0f32; 8];
        unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
        sum += buf.iter().copied().sum::<f32>();
        while i < d {
            sum += (unsafe { *v1.get_unchecked(i) } - unsafe { *v2.get_unchecked(i) }).abs();
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&[N], &[f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_pd() };
        let abs_mask = unsafe { _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffff)) };
        while i + 4 <= d {
            unsafe {
                let a = _mm256_loadu_pd(v1.as_ptr().add(i));
                let b = _mm256_loadu_pd(v2.as_ptr().add(i));
                let diff = _mm256_sub_pd(a, b);
                let abs = _mm256_and_pd(diff, abs_mask);
                acc = _mm256_add_pd(acc, abs);
            }
            i += 4;
        }
        let mut sum: f64 = 0.0;
        let mut buf = [0f64; 4];
        unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
        sum += buf.iter().copied().sum::<f64>();
        while i < d {
            sum += (unsafe { *v1.get_unchecked(i) } - unsafe { *v2.get_unchecked(i) }).abs();
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    super::scalar::l1dist(v1, v2, d)
}

#[inline(always)]
pub(super) fn mul<N>(v1: &mut [N], v2: &[N], a: N, d: usize)
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let scalar = unsafe { _mm256_set1_ps(a.to_f32().unwrap()) };
        let mut i = 0;
        while i + 8 <= d {
            unsafe {
                let b = _mm256_loadu_ps(v2.as_ptr().add(i));
                let prod = _mm256_mul_ps(b, scalar);
                _mm256_storeu_ps(v1.as_mut_ptr().add(i), prod);
            }
            i += 8;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a.to_f32().unwrap();
            }
            i += 1;
        }
        return;
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let scalar = unsafe { _mm256_set1_pd(a.to_f64().unwrap()) };
        let mut i = 0;
        while i + 4 <= d {
            unsafe {
                let b = _mm256_loadu_pd(v2.as_ptr().add(i));
                let prod = _mm256_mul_pd(b, scalar);
                _mm256_storeu_pd(v1.as_mut_ptr().add(i), prod);
            }
            i += 4;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) = *v2.get_unchecked(i) * a.to_f64().unwrap();
            }
            i += 1;
        }
        return;
    }
    super::scalar::mul(v1, v2, a, d)
}

#[inline(always)]
pub(super) fn mul_assign<N>(v: &mut [N], f: N, d: usize)
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v = unsafe { std::mem::transmute::<&mut [N], &mut [f32]>(v) };
        let scalar = unsafe { _mm256_set1_ps(f.to_f32().unwrap()) };
        let mut i = 0;
        while i + 8 <= d {
            unsafe {
                let ptr = v.as_mut_ptr().add(i);
                let mut val = _mm256_loadu_ps(ptr);
                val = _mm256_mul_ps(val, scalar);
                _mm256_storeu_ps(ptr, val);
            }
            i += 8;
        }
        while i < d {
            unsafe {
                *v.get_unchecked_mut(i) *= f.to_f32().unwrap();
            }
            i += 1;
        }
        return;
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v = unsafe { std::mem::transmute::<&mut [N], &mut [f64]>(v) };
        let scalar = unsafe { _mm256_set1_pd(f.to_f64().unwrap()) };
        let mut i = 0;
        while i + 4 <= d {
            unsafe {
                let ptr = v.as_mut_ptr().add(i);
                let mut val = _mm256_loadu_pd(ptr);
                val = _mm256_mul_pd(val, scalar);
                _mm256_storeu_pd(ptr, val);
            }
            i += 4;
        }
        while i < d {
            unsafe {
                *v.get_unchecked_mut(i) *= f.to_f64().unwrap();
            }
            i += 1;
        }
        return;
    }
    super::scalar::mul_assign(v, f, d)
}

#[inline(always)]
pub(super) fn add_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let mut i = 0;
        while i + 8 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let ptr2 = v2.as_ptr().add(i);
                let a = _mm256_loadu_ps(ptr1);
                let b = _mm256_loadu_ps(ptr2);
                let sum = _mm256_add_ps(a, b);
                _mm256_storeu_ps(ptr1, sum);
            }
            i += 8;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
            }
            i += 1;
        }
        return;
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let mut i = 0;
        while i + 4 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let ptr2 = v2.as_ptr().add(i);
                let a = _mm256_loadu_pd(ptr1);
                let b = _mm256_loadu_pd(ptr2);
                let sum = _mm256_add_pd(a, b);
                _mm256_storeu_pd(ptr1, sum);
            }
            i += 4;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i);
            }
            i += 1;
        }
        return;
    }
    super::scalar::add_assign(v1, v2, d)
}

#[inline(always)]
pub(super) fn sub_assign<N>(v1: &mut [N], v2: &[N], d: usize)
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let mut i = 0;
        while i + 8 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let ptr2 = v2.as_ptr().add(i);
                let a = _mm256_loadu_ps(ptr1);
                let b = _mm256_loadu_ps(ptr2);
                let diff = _mm256_sub_ps(a, b);
                _mm256_storeu_ps(ptr1, diff);
            }
            i += 8;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
            }
            i += 1;
        }
        return;
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let mut i = 0;
        while i + 4 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let ptr2 = v2.as_ptr().add(i);
                let a = _mm256_loadu_pd(ptr1);
                let b = _mm256_loadu_pd(ptr2);
                let diff = _mm256_sub_pd(a, b);
                _mm256_storeu_pd(ptr1, diff);
            }
            i += 4;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) -= *v2.get_unchecked(i);
            }
            i += 1;
        }
        return;
    }
    super::scalar::sub_assign(v1, v2, d)
}

#[inline(always)]
pub(super) fn fmamul<N>(v1: &mut [N], a: N, v2: &[N], b: N, d: usize)
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let af = a.to_f32().unwrap();
        let bf = b.to_f32().unwrap();
        let va = unsafe { _mm256_set1_ps(af) };
        let vb = unsafe { _mm256_set1_ps(bf) };
        let mut i = 0;
        while i + 8 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let ptr2 = v2.as_ptr().add(i);
                let mut x = _mm256_loadu_ps(ptr1);
                let y = _mm256_loadu_ps(ptr2);
                x = _mm256_fmadd_ps(x, va, y);
                x = _mm256_mul_ps(x, vb);
                _mm256_storeu_ps(ptr1, x);
            }
            i += 8;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) =
                    v1.get_unchecked(i).mul_add(af, *v2.get_unchecked(i)) * bf;
            }
            i += 1;
        }
        return;
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let af = a.to_f64().unwrap();
        let bf = b.to_f64().unwrap();
        let va = unsafe { _mm256_set1_pd(af) };
        let vb = unsafe { _mm256_set1_pd(bf) };
        let mut i = 0;
        while i + 4 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let ptr2 = v2.as_ptr().add(i);
                let mut x = _mm256_loadu_pd(ptr1);
                let y = _mm256_loadu_pd(ptr2);
                x = _mm256_fmadd_pd(x, va, y);
                x = _mm256_mul_pd(x, vb);
                _mm256_storeu_pd(ptr1, x);
            }
            i += 4;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) =
                    v1.get_unchecked(i).mul_add(af, *v2.get_unchecked(i)) * bf;
            }
            i += 1;
        }
        return;
    }
    super::scalar::fmamul(v1, a, v2, b, d)
}

#[inline(always)]
pub(super) fn dot<N>(v1: &[N], v2: &[N], d: usize) -> N
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&[N], &[f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_ps() };
        while i + 8 <= d {
            unsafe {
                let a = _mm256_loadu_ps(v1.as_ptr().add(i));
                let b = _mm256_loadu_ps(v2.as_ptr().add(i));
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            i += 8;
        }
        let mut sum: f32 = 0.0;
        let mut buf = [0f32; 8];
        unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
        sum += buf.iter().copied().sum::<f32>();
        while i < d {
            sum += unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) };
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&[N], &[f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_pd() };
        while i + 4 <= d {
            unsafe {
                let a = _mm256_loadu_pd(v1.as_ptr().add(i));
                let b = _mm256_loadu_pd(v2.as_ptr().add(i));
                acc = _mm256_fmadd_pd(a, b, acc);
            }
            i += 4;
        }
        let mut sum: f64 = 0.0;
        let mut buf = [0f64; 4];
        unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
        sum += buf.iter().copied().sum::<f64>();
        while i < d {
            sum += unsafe { *v1.get_unchecked(i) * *v2.get_unchecked(i) };
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    super::scalar::dot(v1, v2, d)
}

#[inline(always)]
pub(super) fn axpy<N>(v1: &mut [N], a: N, v2: &[N], d: usize)
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f32]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f32]>(v2) };
        let scalar = a.to_f32().unwrap();
        let mut i = 0;
        while i + 8 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let b = _mm256_loadu_ps(v2.as_ptr().add(i));
                let prod = _mm256_mul_ps(b, _mm256_set1_ps(scalar));
                let orig = _mm256_loadu_ps(ptr1);
                let sum = _mm256_add_ps(orig, prod);
                _mm256_storeu_ps(ptr1, sum);
            }
            i += 8;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) * scalar;
            }
            i += 1;
        }
        return;
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v1 = unsafe { std::mem::transmute::<&mut [N], &mut [f64]>(v1) };
        let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
        let scalar = a.to_f64().unwrap();
        let mut i = 0;
        while i + 4 <= d {
            unsafe {
                let ptr1 = v1.as_mut_ptr().add(i);
                let b = _mm256_loadu_pd(v2.as_ptr().add(i));
                let prod = _mm256_mul_pd(b, _mm256_set1_pd(scalar));
                let orig = _mm256_loadu_pd(ptr1);
                let sum = _mm256_add_pd(orig, prod);
                _mm256_storeu_pd(ptr1, sum);
            }
            i += 4;
        }
        while i < d {
            unsafe {
                *v1.get_unchecked_mut(i) += *v2.get_unchecked(i) * scalar;
            }
            i += 1;
        }
        return;
    }
    super::scalar::axpy(v1, a, v2, d)
}

#[inline(always)]
pub(super) fn sum<N>(v: &[N], d: usize) -> N
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v = unsafe { std::mem::transmute::<&[N], &[f32]>(v) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_ps() };
        while i + 8 <= d {
            unsafe {
                let x = _mm256_loadu_ps(v.as_ptr().add(i));
                acc = _mm256_add_ps(acc, x);
            }
            i += 8;
        }
        let mut buf = [0f32; 8];
        unsafe { _mm256_storeu_ps(buf.as_mut_ptr(), acc) };
        let mut sum: f32 = buf.iter().copied().sum();
        while i < d {
            sum += unsafe { *v.get_unchecked(i) };
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v = unsafe { std::mem::transmute::<&[N], &[f64]>(v) };
        let mut i = 0;
        let mut acc = unsafe { _mm256_setzero_pd() };
        while i + 4 <= d {
            unsafe {
                let x = _mm256_loadu_pd(v.as_ptr().add(i));
                acc = _mm256_add_pd(acc, x);
            }
            i += 4;
        }
        let mut buf = [0f64; 4];
        unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
        let mut sum: f64 = buf.iter().copied().sum();
        while i < d {
            sum += unsafe { *v.get_unchecked(i) };
            i += 1;
        }
        return N::from(sum).unwrap();
    }
    super::scalar::sum(v, d)
}

#[inline(always)]
pub(super) fn add_scalar<N>(v: &mut [N], s: N, d: usize)
where
    N: Float,
{
    if TypeId::of::<N>() == TypeId::of::<f32>() {
        let v = unsafe { std::mem::transmute::<&mut [N], &mut [f32]>(v) };
        let scalar = s.to_f32().unwrap();
        let mut i = 0;
        while i + 8 <= d {
            unsafe {
                let ptr = v.as_mut_ptr().add(i);
                let mut x = _mm256_loadu_ps(ptr);
                x = _mm256_add_ps(x, _mm256_set1_ps(scalar));
                _mm256_storeu_ps(ptr, x);
            }
            i += 8;
        }
        while i < d {
            unsafe {
                *v.get_unchecked_mut(i) += scalar;
            }
            i += 1;
        }
        return;
    }
    if TypeId::of::<N>() == TypeId::of::<f64>() {
        let v = unsafe { std::mem::transmute::<&mut [N], &mut [f64]>(v) };
        let scalar = s.to_f64().unwrap();
        let mut i = 0;
        while i + 4 <= d {
            unsafe {
                let ptr = v.as_mut_ptr().add(i);
                let mut x = _mm256_loadu_pd(ptr);
                x = _mm256_add_pd(x, _mm256_set1_pd(scalar));
                _mm256_storeu_pd(ptr, x);
            }
            i += 4;
        }
        while i < d {
            unsafe {
                *v.get_unchecked_mut(i) += scalar;
            }
            i += 1;
        }
        return;
    }
    super::scalar::add_scalar(v, s, d)
}
