use std::arch::x86_64::*;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign, SubAssign};

use num_traits::Float; //FIXME: use crate::Float instead

use crate::math::{DefaultMath, Math as MathTrait};

/// AVX2‑accelerated implementation.
///
/// This version duplicates the unrolled code, but instead of scalar loops it
/// invokes **explicit AVX2 intrinsics**.  When possible we also use FMA
/// instructions (`_mm256_fmadd_*`); note that the results may therefore
/// differ slightly from the generic math because of differing rounding order.
/// The module is kept separate so comparisons with other backends remain easy
/// and to isolate the unsafe intrinsics.
pub struct AVX2Math<N> {
    phantom: PhantomData<N>,
}

#[cfg(target_arch = "x86_64")]
impl<N> MathTrait<N> for AVX2Math<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + 'static,
{
    #[inline(always)]
    fn sqdist(v1: &[N], v2: &[N], d: usize) -> N {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::sqdist(v1, v2, d)
    }

    #[inline(always)]
    fn l1dist(v1: &[N], v2: &[N], d: usize) -> N {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
            let v1 = unsafe { std::mem::transmute::<&[N], &[f64]>(v1) };
            let v2 = unsafe { std::mem::transmute::<&[N], &[f64]>(v2) };
            let mut i = 0;
            let mut acc = unsafe { _mm256_setzero_pd() };
            let abs_mask =
                unsafe { _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fff_ffff_ffff_ffff)) };
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
        DefaultMath::<N>::l1dist(v1, v2, d)
    }

    #[inline(always)]
    fn mul(v1: &mut [N], v2: &[N], a: N, d: usize) {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::mul(v1, v2, a, d)
    }

    #[inline(always)]
    fn mul_assign(v: &mut [N], f: N, d: usize) {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::mul_assign(v, f, d)
    }

    #[inline(always)]
    fn add_assign(v1: &mut [N], v2: &[N], d: usize) {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::add_assign(v1, v2, d)
    }

    #[inline(always)]
    fn sub_assign(v1: &mut [N], v2: &[N], d: usize) {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::sub_assign(v1, v2, d)
    }

    #[inline(always)]
    fn fmamul(v1: &mut [N], a: N, v2: &[N], b: N, d: usize) {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::fmamul(v1, a, v2, b, d)
    }

    #[inline(always)]
    fn dot(v1: &[N], v2: &[N], d: usize) -> N {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::dot(v1, v2, d)
    }

    #[inline(always)]
    fn axpy(v1: &mut [N], a: N, v2: &[N], d: usize) {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::axpy(v1, a, v2, d)
    }

    #[inline(always)]
    fn sum(v: &[N], d: usize) -> N {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::sum(v, d)
    }

    #[inline(always)]
    fn add_scalar(v: &mut [N], s: N, d: usize) {
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f32>() {
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
        if std::any::TypeId::of::<N>() == std::any::TypeId::of::<f64>() {
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
        DefaultMath::<N>::add_scalar(v, s, d)
    }

    #[inline(always)]
    fn copy(v1: &mut [N], v2: &[N], d: usize)
    where
        N: Copy,
    {
        debug_assert!(v1.len() >= d && v2.len() >= d);
        unsafe {
            std::ptr::copy_nonoverlapping(v2.as_ptr(), v1.as_mut_ptr(), d);
        }
    }

    #[inline(always)]
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
}
