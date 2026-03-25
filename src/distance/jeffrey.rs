use crate::Float;
use crate::distance::DistanceFunction;

pub fn jeffrey_divergence<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let d = a.len().min(b.len());
    let mut sum = F::zero();

    for i in 0..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).to_float::<F>();
            let right: F = (*b.get_unchecked(i)).to_float::<F>();

            if left > F::zero() && right > F::zero() {
                sum = sum + left * (left / right).ln() + right * (right / left).ln();
            }
        }
    }

    sum
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JeffreyDistance;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for JeffreyDistance {
    fn distance(&self, a: &[N], b: &[N]) -> F { jeffrey_divergence(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    #[test]
    fn jeffrey_is_zero_for_identical_vectors() {
        let a = [0.2, 0.3, 0.5];
        approx_eq(jeffrey_divergence::<f64, f64>(&a, &a), 0.0);
    }

    #[test]
    fn jeffrey_is_symmetric() {
        let a = [0.5, 0.5];
        let b = [0.25, 0.75];
        let ab = jeffrey_divergence::<f64, f64>(&a, &b);
        let ba = jeffrey_divergence::<f64, f64>(&b, &a);
        approx_eq(ab, ba);
    }

    #[test]
    fn jeffrey_matches_known_value() {
        let a = [0.5, 0.5];
        let b = [0.25, 0.75];
        approx_eq(jeffrey_divergence::<f64, f64>(&a, &b), 0.27465307216702745);
    }

    #[test]
    fn jeffrey_zero_pairs_do_not_produce_nan() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        approx_eq(jeffrey_divergence::<f64, f64>(&a, &b), 0.0);
    }

    #[test]
    fn jeffrey_returns_zero_for_empty_input() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        approx_eq(jeffrey_divergence::<f64, f64>(&a, &b), 0.0);
    }
}
