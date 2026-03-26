use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _CMP_EQ_OQ, _mm256_add_pd, _mm256_add_ps, _mm256_andnot_pd, _mm256_andnot_ps, _mm256_blendv_pd,
    _mm256_blendv_ps, _mm256_cmp_pd, _mm256_cmp_ps, _mm256_div_pd, _mm256_div_ps, _mm256_loadu_pd,
    _mm256_loadu_ps, _mm256_mul_pd, _mm256_mul_ps, _mm256_set1_pd, _mm256_set1_ps,
    _mm256_setzero_pd, _mm256_setzero_ps, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd,
    _mm256_sub_ps,
};

use crate::Float;
use crate::distance::DistanceFunction;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn clark_distance_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    let d = a.len().min(b.len());
    let sd = d & !7;
    let mut acc = _mm256_setzero_ps();
    let zero = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);

    for i in (0..sd).step_by(8) {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let diff = _mm256_sub_ps(va, vb);
        let numerator = _mm256_andnot_ps(sign_mask, diff);
        let abs_a = _mm256_andnot_ps(sign_mask, va);
        let abs_b = _mm256_andnot_ps(sign_mask, vb);
        let denominator = _mm256_add_ps(abs_a, abs_b);
        let ratio = _mm256_div_ps(numerator, denominator);
        let term = _mm256_mul_ps(ratio, ratio);
        let den_is_zero = _mm256_cmp_ps(denominator, zero, _CMP_EQ_OQ);
        let term = _mm256_blendv_ps(term, zero, den_is_zero);
        acc = _mm256_add_ps(acc, term);
    }

    let mut tmp = [0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f32>();

    for i in sd..d {
        unsafe {
            let left = *a.get_unchecked(i);
            let right = *b.get_unchecked(i);
            let numerator = (left - right).abs();
            let denominator = left.abs() + right.abs();
            if denominator != 0.0 {
                let ratio = numerator / denominator;
                sum += ratio * ratio;
            }
        }
    }
    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn clark_distance_f64_avx(a: &[f64], b: &[f64]) -> f64 {
    let d = a.len().min(b.len());
    let sd = d & !3;
    let mut acc = _mm256_setzero_pd();
    let zero = _mm256_setzero_pd();
    let sign_mask = _mm256_set1_pd(-0.0);

    for i in (0..sd).step_by(4) {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        let diff = _mm256_sub_pd(va, vb);
        let numerator = _mm256_andnot_pd(sign_mask, diff);
        let abs_a = _mm256_andnot_pd(sign_mask, va);
        let abs_b = _mm256_andnot_pd(sign_mask, vb);
        let denominator = _mm256_add_pd(abs_a, abs_b);
        let ratio = _mm256_div_pd(numerator, denominator);
        let term = _mm256_mul_pd(ratio, ratio);
        let den_is_zero = _mm256_cmp_pd(denominator, zero, _CMP_EQ_OQ);
        let term = _mm256_blendv_pd(term, zero, den_is_zero);
        acc = _mm256_add_pd(acc, term);
    }

    let mut tmp = [0f64; 4];
    unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f64>();

    for i in sd..d {
        unsafe {
            let left = *a.get_unchecked(i);
            let right = *b.get_unchecked(i);
            let numerator = (left - right).abs();
            let denominator = left.abs() + right.abs();
            if denominator != 0.0 {
                let ratio = numerator / denominator;
                sum += ratio * ratio;
            }
        }
    }
    sum.sqrt()
}

/// Clark distance:
/// $$d_{CL}(a,b)=\sqrt{\sum_i \left(\frac{a_i-b_i}{a_i+b_i}\right)^2}$$
pub fn clark_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") {
        if TypeId::of::<N>() == TypeId::of::<f32>() && TypeId::of::<F>() == TypeId::of::<f32>() {
            unsafe {
                let a_f32 = &*(a as *const [N] as *const [f32]);
                let b_f32 = &*(b as *const [N] as *const [f32]);
                let result = clark_distance_f32_avx(a_f32, b_f32);
                return std::mem::transmute_copy::<f32, F>(&result);
            };
        }

        if TypeId::of::<N>() == TypeId::of::<f64>() && TypeId::of::<F>() == TypeId::of::<f64>() {
            unsafe {
                let a_f64 = &*(a as *const [N] as *const [f64]);
                let b_f64 = &*(b as *const [N] as *const [f64]);
                let result = clark_distance_f64_avx(a_f64, b_f64);
                return std::mem::transmute_copy::<f64, F>(&result);
            };
        }
    }
    clark_distance_fallback(a, b)
}

fn clark_distance_fallback<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    const LANES: usize = 8;

    let d = a.len().min(b.len());
    let sd = d & !(LANES - 1);
    let mut vsum = [F::zero(); LANES];

    for i in (0..sd).step_by(LANES) {
        for j in 0..LANES {
            unsafe {
                let left: F = (*a.get_unchecked(i + j)).to_float::<F>();
                let right: F = (*b.get_unchecked(i + j)).to_float::<F>();
                let numerator = (left - right).abs();
                let denominator = left.abs() + right.abs();
                let term = if denominator == F::zero() {
                    F::zero()
                } else {
                    let ratio = numerator / denominator;
                    ratio * ratio
                };
                *vsum.get_unchecked_mut(j) = *vsum.get_unchecked(j) + term;
            }
        }
    }

    let mut sum = vsum.iter().copied().fold(F::zero(), |acc, value| acc + value);

    for i in sd..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).to_float::<F>();
            let right: F = (*b.get_unchecked(i)).to_float::<F>();
            let numerator = (left - right).abs();
            let denominator = left.abs() + right.abs();
            if denominator != F::zero() {
                let ratio = numerator / denominator;
                sum = sum + ratio * ratio;
            }
        }
    }

    sum.sqrt()
}

#[derive(Debug, Clone, Copy, Default)]
/// Clark distance strategy (square-root scaled L2 over sum).
pub struct Clark;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for Clark {
    fn distance(&self, a: &[N], b: &[N]) -> F { clark_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}
