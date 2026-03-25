use super::DistanceFunction;
use crate::Float;

pub fn chi_squared_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let d = a.len().min(b.len());
    let mut sum = F::zero();

    for i in 0..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).to_float::<F>();
            let right: F = (*b.get_unchecked(i)).to_float::<F>();
            let denominator = left + right;

            if denominator != F::zero() {
                let diff = left - right;
                sum = sum + (diff * diff) / denominator;
            }
        }
    }

    sum
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChiSquaredDistance;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for ChiSquaredDistance {
    fn distance(&self, a: &[N], b: &[N]) -> F { chi_squared_distance(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    #[test]
    fn chi_squared_is_zero_for_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        approx_eq(chi_squared_distance::<f64, f64>(&a, &a), 0.0);
    }

    #[test]
    fn chi_squared_matches_known_value() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        approx_eq(chi_squared_distance::<f64, f64>(&a, &b), 5.0 / 3.0);
    }

    #[test]
    fn chi_squared_skips_zero_denominator_term() {
        let a = [0.0, 1.0];
        let b = [0.0, 2.0];
        approx_eq(chi_squared_distance::<f64, f64>(&a, &b), 1.0 / 3.0);
    }

    #[test]
    fn chi_squared_returns_zero_for_empty_input() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        approx_eq(chi_squared_distance::<f64, f64>(&a, &b), 0.0);
    }
}
