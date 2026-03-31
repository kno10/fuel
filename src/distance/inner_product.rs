//! Right now, we do not yet have a similarity function trait
//! So this only exposes an optimized dot product function.

use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_setzero_pd,
    _mm256_setzero_ps, _mm256_storeu_pd, _mm256_storeu_ps,
};

use crate::Float;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn dot_f32_avx_fma(a: &[f32], b: &[f32]) -> f32 {
    let d = a.len().min(b.len());
    let sd = d & !7;

    let mut acc = _mm256_setzero_ps();
    for i in (0..sd).step_by(8) {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut tmp = [0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f32>();

    for i in sd..d {
        sum += a[i] * b[i];
    }

    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn dot_f64_avx_fma(a: &[f64], b: &[f64]) -> f64 {
    let d = a.len().min(b.len());
    let sd = d & !3;

    let mut acc = _mm256_setzero_pd();
    for i in (0..sd).step_by(4) {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        acc = _mm256_fmadd_pd(va, vb, acc);
    }

    let mut tmp = [0f64; 4];
    unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f64>();

    for i in sd..d {
        sum += a[i] * b[i];
    }

    sum
}

fn dot_fallback<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let d = a.len().min(b.len());
    let mut sum = F::zero();
    for i in 0..d {
        let left: F = a[i].to_float::<F>();
        let right: F = b[i].to_float::<F>();
        sum += left * right;
    }
    sum
}

/// Inner product (dot product) between vectors.
///
/// This implementation uses optimized AVX+FMA code for f32/f64 on x86_64 when available,
/// otherwise falls back to scalar computation.
pub fn dot<N: Float + 'static, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        if TypeId::of::<N>() == TypeId::of::<f32>() && TypeId::of::<F>() == TypeId::of::<f32>() {
            unsafe {
                let a_f32 = &*(a as *const [N] as *const [f32]);
                let b_f32 = &*(b as *const [N] as *const [f32]);
                let result = dot_f32_avx_fma(a_f32, b_f32);
                return result.to_float::<F>();
            }
        }

        if TypeId::of::<N>() == TypeId::of::<f64>() && TypeId::of::<F>() == TypeId::of::<f64>() {
            unsafe {
                let a_f64 = &*(a as *const [N] as *const [f64]);
                let b_f64 = &*(b as *const [N] as *const [f64]);
                let result = dot_f64_avx_fma(a_f64, b_f64);
                return result.to_float::<F>();
            }
        }
    }

    dot_fallback(a, b)
}
