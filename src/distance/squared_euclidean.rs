use std::any::TypeId;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm256_fmadd_pd, _mm256_fmadd_ps, _mm256_loadu_pd, _mm256_loadu_ps, _mm256_setzero_pd,
    _mm256_setzero_ps, _mm256_storeu_pd, _mm256_storeu_ps, _mm256_sub_pd, _mm256_sub_ps,
};

use crate::Float;
use crate::distance::DistanceFunction;
use crate::distance::partial::PartialDistance;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn squared_euclidean_distance_f32_avx_fma(a: &[f32], b: &[f32]) -> f32 {
    let d = a.len().min(b.len());
    let sd = d & !7;
    let mut acc = _mm256_setzero_ps();

    for i in (0..sd).step_by(8) {
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    let mut tmp = [0f32; 8];
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f32>();

    for i in sd..d {
        unsafe {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum = diff.mul_add(diff, sum);
        }
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn squared_euclidean_distance_f64_avx_fma(a: &[f64], b: &[f64]) -> f64 {
    let d = a.len().min(b.len());
    let sd = d & !3;
    let mut acc = _mm256_setzero_pd();

    for i in (0..sd).step_by(4) {
        let va = unsafe { _mm256_loadu_pd(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_pd(b.as_ptr().add(i)) };
        let diff = _mm256_sub_pd(va, vb);
        acc = _mm256_fmadd_pd(diff, diff, acc);
    }

    let mut tmp = [0f64; 4];
    unsafe { _mm256_storeu_pd(tmp.as_mut_ptr(), acc) };
    let mut sum = tmp.iter().copied().sum::<f64>();

    for i in sd..d {
        unsafe {
            let diff = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum = diff.mul_add(diff, sum);
        }
    }
    sum
}

/// Squared Euclidean distance:
/// $$d_2^2(a,b)=\sum_i (a_i-b_i)^2$$
pub fn squared_euclidean_distance<N, F>(a: &[N], b: &[N]) -> F
where
    N: Float + 'static,
    F: Float + 'static,
{
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
        if TypeId::of::<N>() == TypeId::of::<f32>() && TypeId::of::<F>() == TypeId::of::<f32>() {
            unsafe {
                let a_f32 = &*(a as *const [N] as *const [f32]);
                let b_f32 = &*(b as *const [N] as *const [f32]);
                let result = squared_euclidean_distance_f32_avx_fma(a_f32, b_f32);
                return std::mem::transmute_copy::<f32, F>(&result);
            }
        }

        if TypeId::of::<N>() == TypeId::of::<f64>() && TypeId::of::<F>() == TypeId::of::<f64>() {
            unsafe {
                let a_f64 = &*(a as *const [N] as *const [f64]);
                let b_f64 = &*(b as *const [N] as *const [f64]);
                let result = squared_euclidean_distance_f64_avx_fma(a_f64, b_f64);
                return std::mem::transmute_copy::<f64, F>(&result);
            }
        }
    }
    squared_euclidean_distance_fallback(a, b)
}

fn squared_euclidean_distance_fallback<N, F>(a: &[N], b: &[N]) -> F
where
    N: Float + 'static,
    F: Float + 'static,
{
    const LANES: usize = 8;

    let d = a.len().min(b.len());
    let sd = d & !(LANES - 1);
    let mut vsum = [N::zero(); LANES];

    for i in (0..sd).step_by(LANES) {
        for j in 0..LANES {
            unsafe {
                let left = *a.get_unchecked(i + j);
                let right = *b.get_unchecked(i + j);
                let diff = left - right;
                *vsum.get_unchecked_mut(j) = diff * diff + *vsum.get_unchecked(j);
            }
        }
    }

    let mut sum_n = vsum.iter().copied().fold(N::zero(), |acc, value| acc + value);

    for i in sd..d {
        unsafe {
            let left = *a.get_unchecked(i);
            let right = *b.get_unchecked(i);
            let diff = left - right;
            sum_n = diff * diff + sum_n;
        }
    }

    sum_n.to_float::<F>()
}

#[derive(Debug, Clone, Copy, Default)]
/// Squared Euclidean distance strategy (L2 squared).
pub struct SquaredEuclidean;

impl<N, F> DistanceFunction<[N], F> for SquaredEuclidean
where
    N: Float + 'static,
    F: Float + 'static,
{
    fn distance(&self, a: &[N], b: &[N]) -> F { squared_euclidean_distance(a, b) }
}

impl<N, F> DistanceFunction<Vec<N>, F> for SquaredEuclidean
where
    N: Float + 'static,
    F: Float + 'static,
{
    fn distance(&self, a: &Vec<N>, b: &Vec<N>) -> F { squared_euclidean_distance(a, b) }
}

impl<N, F> PartialDistance<N, F> for SquaredEuclidean
where
    N: Float + 'static,
    F: Float + 'static,
{
    fn axis_distance(&self, delta: N) -> F {
        let delta_f: F = delta.to_float::<F>();
        delta_f * delta_f
    }

    fn distance_to_range_bound(&self, distance: F) -> F { distance }

    fn range_bound_to_distance(&self, bound: F) -> F { bound }

    fn replace_axis_distance(
        &self, current: F, _axis: usize, old_axis: F, new_axis: F, _axis_bounds: &[F],
    ) -> F {
        // Squared Euclidean is additive.
        current - old_axis + new_axis
    }
}
