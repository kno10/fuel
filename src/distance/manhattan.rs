use num_traits::{AsPrimitive, Float, ToPrimitive};
use std::any::TypeId;

use super::{DistanceFunction, DistanceMetric};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_add_pd, _mm256_add_ps, _mm256_andnot_pd, _mm256_andnot_ps, _mm256_loadu_pd,
    _mm256_loadu_ps, _mm256_set1_pd, _mm256_set1_ps, _mm256_setzero_pd, _mm256_setzero_ps,
    _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps,
};

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn manhattan_distance_f32_avx(a: &[f32], b: &[f32]) -> f32 {
    let d = a.len().min(b.len());
    let sd = d & !7;
    let mut acc = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0);

    for i in (0..sd).step_by(8) {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let diff = _mm256_sub_ps(va, vb);
        let abs = _mm256_andnot_ps(sign_mask, diff);
        acc = _mm256_add_ps(acc, abs);
    }

    let mut tmp = [0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f32>();

    for i in sd..d {
        unsafe {
            sum += (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn manhattan_distance_f64_avx(a: &[f64], b: &[f64]) -> f64 {
    let d = a.len().min(b.len());
    let sd = d & !3;
    let mut acc = _mm256_setzero_pd();
    let sign_mask = _mm256_set1_pd(-0.0);

    for i in (0..sd).step_by(4) {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        let diff = _mm256_sub_pd(va, vb);
        let abs = _mm256_andnot_pd(sign_mask, diff);
        acc = _mm256_add_pd(acc, abs);
    }

    let mut tmp = [0f64; 4];
    unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f64>();

    for i in sd..d {
        unsafe {
            sum += (*a.get_unchecked(i) - *b.get_unchecked(i)).abs();
        }
    }
    sum
}

pub fn manhattan_distance<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") {
        if TypeId::of::<N>() == TypeId::of::<f32>() && TypeId::of::<F>() == TypeId::of::<f32>() {
            unsafe {
                let a_f32 = &*(a as *const [N] as *const [f32]);
                let b_f32 = &*(b as *const [N] as *const [f32]);
                let result = manhattan_distance_f32_avx(a_f32, b_f32);
                return std::mem::transmute_copy::<f32, F>(&result);
            };
        }

        if TypeId::of::<N>() == TypeId::of::<f64>() && TypeId::of::<F>() == TypeId::of::<f64>() {
            unsafe {
                let a_f64 = &*(a as *const [N] as *const [f64]);
                let b_f64 = &*(b as *const [N] as *const [f64]);
                let result = manhattan_distance_f64_avx(a_f64, b_f64);
                return std::mem::transmute_copy::<f64, F>(&result);
            };
        }
    }
    manhattan_distance_fallback(a, b)
}

fn manhattan_distance_fallback<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    const LANES: usize = 8;

    let d = a.len().min(b.len());
    let sd = d & !(LANES - 1);
    let mut vsum = [F::zero(); LANES];

    for i in (0..sd).step_by(LANES) {
        for j in 0..LANES {
            unsafe {
                let left: F = (*a.get_unchecked(i + j)).as_();
                let right: F = (*b.get_unchecked(i + j)).as_();
                let diff = (left - right).abs();
                *vsum.get_unchecked_mut(j) = *vsum.get_unchecked(j) + diff;
            }
        }
    }

    let mut sum = vsum
        .iter()
        .copied()
        .fold(F::zero(), |acc, value| acc + value);

    for i in sd..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).as_();
            let right: F = (*b.get_unchecked(i)).as_();
            sum = sum + (left - right).abs();
        }
    }

    sum
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ManhattanDistance;

impl<N: Float + ToPrimitive + AsPrimitive<f64>> DistanceMetric<[N]> for ManhattanDistance {}

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceFunction<[N], F>
    for ManhattanDistance
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        manhattan_distance(a, b)
    }
}
