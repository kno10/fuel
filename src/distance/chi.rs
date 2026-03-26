use crate::Float;
use crate::distance::{DistanceFunction, chi_squared_distance};

/// Chi distance:
/// $$d_{\chi}(a,b)=\sqrt{\sum_i \frac{(a_i-b_i)^2}{a_i+b_i}}$$
pub fn chi_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    chi_squared_distance::<N, F>(a, b).sqrt()
}

#[derive(Debug, Clone, Copy, Default)]
/// Chi distance strategy (a scaled chi-square formulation).
pub struct Chi;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for Chi {
    fn distance(&self, a: &[N], b: &[N]) -> F { chi_distance(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    #[test]
    fn chi_is_zero_for_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        approx_eq(chi_distance::<f64, f64>(&a, &a), 0.0);
    }

    #[test]
    fn chi_matches_sqrt_of_chi_squared() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        let expected = (5.0 / 3.0_f64).sqrt();
        approx_eq(chi_distance::<f64, f64>(&a, &b), expected);
    }

    #[test]
    fn chi_returns_zero_for_empty_input() {
        let a: [f64; 0] = [];
        let b: [f64; 0] = [];
        approx_eq(chi_distance::<f64, f64>(&a, &b), 0.0);
    }
}
