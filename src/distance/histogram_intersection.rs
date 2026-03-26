use crate::Float;
use crate::distance::DistanceFunction;

/// Histogram intersection distance:
/// $$d_{HI}=1-\frac{\sum_i\min(a_i,b_i)}{\sum_i a_i}$$
pub fn histogram_intersection_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let d = a.len().min(b.len());
    let mut intersection = F::zero();
    let mut union = F::zero();

    for i in 0..d {
        unsafe {
            let left: F = (*a.get_unchecked(i)).to_float::<F>();
            let right: F = (*b.get_unchecked(i)).to_float::<F>();
            intersection = intersection + left.min(right);
            union = union + left.max(right);
        }
    }

    if union == F::zero() { F::zero() } else { F::one() - intersection / union }
}

#[derive(Debug, Clone, Copy, Default)]
/// Histogram intersection distance strategy.
pub struct HistogramIntersection;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for HistogramIntersection {
    fn distance(&self, a: &[N], b: &[N]) -> F { histogram_intersection_distance(a, b) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "left={left}, right={right}");
    }

    #[test]
    fn histogram_intersection_is_zero_for_identical_vectors() {
        let a = [1.0, 2.0, 3.0];
        approx_eq(histogram_intersection_distance::<f64, f64>(&a, &a), 0.0);
    }

    #[test]
    fn histogram_intersection_is_one_for_disjoint_histograms() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        approx_eq(histogram_intersection_distance::<f64, f64>(&a, &b), 1.0);
    }

    #[test]
    fn histogram_intersection_matches_known_partial_overlap() {
        let a = [1.0, 2.0];
        let b = [2.0, 1.0];
        approx_eq(histogram_intersection_distance::<f64, f64>(&a, &b), 0.5);
    }

    #[test]
    fn histogram_intersection_zero_union_returns_zero() {
        let a = [0.0, 0.0];
        let b = [0.0, 0.0];
        approx_eq(histogram_intersection_distance::<f64, f64>(&a, &b), 0.0);
    }
}
