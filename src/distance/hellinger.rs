use super::DistanceFunction;
use crate::Float;

pub fn hellinger_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let d = a.len().min(b.len());
    let mut sum = F::zero();
    let half = F::one() / (F::one() + F::one());

    for i in 0..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).to_float::<F>();
            let right: F = (*b.get_unchecked(i)).to_float::<F>();
            let left_sqrt = left.max(F::zero()).sqrt();
            let right_sqrt = right.max(F::zero()).sqrt();
            let diff = left_sqrt - right_sqrt;
            sum = sum + diff * diff;
        }
    }

    (half * sum).sqrt()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HellingerDistance;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for HellingerDistance {
    fn distance(&self, a: &[N], b: &[N]) -> F { hellinger_distance(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    #[test]
    fn hellinger_is_zero_for_identical_vectors() {
        let a = [0.2, 0.3, 0.5];
        approx_eq(hellinger_distance::<f64, f64>(&a, &a), 0.0);
    }

    #[test]
    fn hellinger_one_hot_is_one() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        approx_eq(hellinger_distance::<f64, f64>(&a, &b), 1.0);
    }

    #[test]
    fn hellinger_clamps_negative_values_to_zero_before_sqrt() {
        let a = [-1.0, 1.0];
        let b = [1.0, 0.0];
        approx_eq(hellinger_distance::<f64, f64>(&a, &b), 1.0);
    }

    #[test]
    fn hellinger_returns_zero_for_empty_input() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        approx_eq(hellinger_distance::<f64, f64>(&a, &b), 0.0);
    }
}
