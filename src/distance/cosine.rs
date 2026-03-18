use num_traits::{AsPrimitive, Float, ToPrimitive};
use std::any::TypeId;

use super::{DistanceFunction, DistanceMetric};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_setzero_pd,
    _mm256_setzero_ps, _mm256_storeu_pd, _mm256_storeu_ps,
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn cosine_similarity_f32_avx_fma(a: &[f32], b: &[f32]) -> f32 {
    let d = a.len().min(b.len());
    let sd = d & !7;

    let mut dot_acc = _mm256_setzero_ps();
    let mut norm_a_acc = _mm256_setzero_ps();
    let mut norm_b_acc = _mm256_setzero_ps();

    for i in (0..sd).step_by(8) {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };

        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        norm_a_acc = _mm256_fmadd_ps(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_ps(vb, vb, norm_b_acc);
    }

    let mut dot_tmp = [0f32; 8];
    let mut norm_a_tmp = [0f32; 8];
    let mut norm_b_tmp = [0f32; 8];
    unsafe {
        _mm256_storeu_ps(dot_tmp.as_mut_ptr(), dot_acc);
        _mm256_storeu_ps(norm_a_tmp.as_mut_ptr(), norm_a_acc);
        _mm256_storeu_ps(norm_b_tmp.as_mut_ptr(), norm_b_acc);
    }

    let mut dot = dot_tmp.iter().copied().sum::<f32>();
    let mut norm_a = norm_a_tmp.iter().copied().sum::<f32>();
    let mut norm_b = norm_b_tmp.iter().copied().sum::<f32>();

    for i in sd..d {
        unsafe {
            let left = *a.get_unchecked(i);
            let right = *b.get_unchecked(i);
            dot += left * right;
            norm_a += left * left;
            norm_b += right * right;
        }
    }

    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == 0.0 {
        1.0
    } else {
        dot / denominator
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
unsafe fn cosine_similarity_f64_avx_fma(a: &[f64], b: &[f64]) -> f64 {
    let d = a.len().min(b.len());
    let sd = d & !3;

    let mut dot_acc = _mm256_setzero_pd();
    let mut norm_a_acc = _mm256_setzero_pd();
    let mut norm_b_acc = _mm256_setzero_pd();

    for i in (0..sd).step_by(4) {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };

        dot_acc = _mm256_fmadd_pd(va, vb, dot_acc);
        norm_a_acc = _mm256_fmadd_pd(va, va, norm_a_acc);
        norm_b_acc = _mm256_fmadd_pd(vb, vb, norm_b_acc);
    }

    let mut dot_tmp = [0f64; 4];
    let mut norm_a_tmp = [0f64; 4];
    let mut norm_b_tmp = [0f64; 4];
    unsafe {
        _mm256_storeu_pd(dot_tmp.as_mut_ptr(), dot_acc);
        _mm256_storeu_pd(norm_a_tmp.as_mut_ptr(), norm_a_acc);
        _mm256_storeu_pd(norm_b_tmp.as_mut_ptr(), norm_b_acc);
    }

    let mut dot = dot_tmp.iter().copied().sum::<f64>();
    let mut norm_a = norm_a_tmp.iter().copied().sum::<f64>();
    let mut norm_b = norm_b_tmp.iter().copied().sum::<f64>();

    for i in sd..d {
        unsafe {
            let left = *a.get_unchecked(i);
            let right = *b.get_unchecked(i);
            dot += left * right;
            norm_a += left * left;
            norm_b += right * right;
        }
    }

    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == 0.0 {
        1.0
    } else {
        dot / denominator
    }
}

fn cosine_similarity<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        if TypeId::of::<N>() == TypeId::of::<f32>() && TypeId::of::<F>() == TypeId::of::<f32>() {
            unsafe {
                let a_f32 = &*(a as *const [N] as *const [f32]);
                let b_f32 = &*(b as *const [N] as *const [f32]);
                let result = cosine_similarity_f32_avx_fma(a_f32, b_f32);
                return std::mem::transmute_copy::<f32, F>(&result);
            }
        }

        if TypeId::of::<N>() == TypeId::of::<f64>() && TypeId::of::<F>() == TypeId::of::<f64>() {
            unsafe {
                let a_f64 = &*(a as *const [N] as *const [f64]);
                let b_f64 = &*(b as *const [N] as *const [f64]);
                let result = cosine_similarity_f64_avx_fma(a_f64, b_f64);
                return std::mem::transmute_copy::<f64, F>(&result);
            }
        }
    }

    cosine_similarity_fallback(a, b)
}

fn cosine_similarity_fallback<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    let d = a.len().min(b.len());
    let mut dot = F::zero();
    let mut norm_a = F::zero();
    let mut norm_b = F::zero();

    for i in 0..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).as_();
            let right: F = (*b.get_unchecked(i)).as_();
            dot = dot + left * right;
            norm_a = norm_a + left * left;
            norm_b = norm_b + right * right;
        }
    }

    let denominator = norm_a.sqrt() * norm_b.sqrt();
    if denominator == F::zero() {
        F::one()
    } else {
        dot / denominator
    }
}

pub fn cosine_distance<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    F::one() - cosine_similarity(a, b)
}

pub fn arccosine_distance<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    cosine_similarity(a, b).max(-F::one()).min(F::one()).acos()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CosineDistance;

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceFunction<[N], F>
    for CosineDistance
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        cosine_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ArccosineDistance;

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceMetric<[N], F>
    for ArccosineDistance
{
}

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceFunction<[N], F>
    for ArccosineDistance
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        arccosine_distance(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    #[test]
    fn cosine_and_arccosine_are_zero_for_same_vector() {
        let a = [1.0, 2.0, 3.0];
        approx_eq(cosine_distance::<f64, f64>(&a, &a), 0.0);
        approx_eq(arccosine_distance::<f64, f64>(&a, &a), 0.0);
    }

    #[test]
    fn orthogonal_vectors_have_expected_distances() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        approx_eq(cosine_distance::<f64, f64>(&a, &b), 1.0);
        approx_eq(
            arccosine_distance::<f64, f64>(&a, &b),
            std::f64::consts::FRAC_PI_2,
        );
    }

    #[test]
    fn opposite_vectors_have_expected_distances() {
        let a = [1.0, 0.0];
        let b = [-1.0, 0.0];
        approx_eq(cosine_distance::<f64, f64>(&a, &b), 2.0);
        approx_eq(arccosine_distance::<f64, f64>(&a, &b), std::f64::consts::PI);
    }

    #[test]
    fn zero_vectors_follow_defined_behavior() {
        let a = [0.0, 0.0];
        let b = [0.0, 0.0];
        approx_eq(cosine_distance::<f64, f64>(&a, &b), 0.0);
        approx_eq(arccosine_distance::<f64, f64>(&a, &b), 0.0);
    }
}
