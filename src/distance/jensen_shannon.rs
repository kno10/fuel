use crate::Float;
use crate::distance::DistanceFunction;

pub fn jensen_shannon_divergence<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let d = a.len().min(b.len());
    let mut sum = F::zero();
    let half = F::one() / (F::one() + F::one());

    for i in 0..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).to_float::<F>();
            let right: F = (*b.get_unchecked(i)).to_float::<F>();
            let mean = (left + right) * half;

            if left > F::zero() && mean > F::zero() {
                sum += left * (left / mean).ln();
            }

            if right > F::zero() && mean > F::zero() {
                sum += right * (right / mean).ln();
            }
        }
    }

    sum * half
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JensenShannon;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for JensenShannon {
    fn distance(&self, a: &[N], b: &[N]) -> F { jensen_shannon_divergence(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    #[test]
    fn jensen_shannon_is_zero_for_identical_vectors() {
        let a = [0.1, 0.3, 0.6];
        approx_eq(jensen_shannon_divergence::<f64, f64>(&a, &a), 0.0);
    }

    #[test]
    fn jensen_shannon_is_symmetric() {
        let a = [0.5, 0.5];
        let b = [0.25, 0.75];
        let ab = jensen_shannon_divergence::<f64, f64>(&a, &b);
        let ba = jensen_shannon_divergence::<f64, f64>(&b, &a);
        approx_eq(ab, ba);
    }

    #[test]
    fn jensen_shannon_one_hot_is_ln2() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        approx_eq(jensen_shannon_divergence::<f64, f64>(&a, &b), std::f64::consts::LN_2);
    }

    #[test]
    fn jensen_shannon_returns_zero_for_empty_input() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        approx_eq(jensen_shannon_divergence::<f64, f64>(&a, &b), 0.0);
    }
}
