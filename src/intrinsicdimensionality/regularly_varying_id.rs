use crate::Float;
use crate::intrinsicdimensionality::{DistanceIDEstimator, find_begin};

/// Regularly varying functions estimator of intrinsic dimensionality.
///
/// Reference:
///
/// L. Amsaleg, O. Chelly, T. Furon, S. Girard, M. E. Houle, K. Kawarabayashi, M. Nett
/// Estimating Local Intrinsic Dimensionality
/// Proc. SIGKDD International Conference on Knowledge Discovery and Data Mining 2015
///
/// Based on the regular variation model and estimates a scaling exponent p using
/// three quantile points (n/2, 3n/4, n), then derives the ID from log-ratios in the tail.
///
/// Returns `NaN` if data are invalid or too few points are available.
pub fn regularly_varying_id<F: Float>(distances: &[F]) -> f64 {
    let begin = find_begin(distances);

    let k = distances.len() - begin;
    if k < 2 {
        return f64::NAN;
    }

    let (n1, n2, n3) = (k >> 1, (3 * k) >> 2, k);
    if n1 < 1 || n2 < 1 || n3 < 1 || begin + n3 > distances.len() {
        return f64::NAN;
    }

    let r1 = distances[begin + n1 - 1];
    let r2 = distances[begin + n2 - 1];
    let r3 = distances[begin + n3 - 1];
    if r1.is_nan()
        || r2.is_nan()
        || r3.is_nan()
        || r1 <= F::zero()
        || r2 <= F::zero()
        || r3 <= F::zero()
    {
        return f64::NAN;
    }
    let r1 = r1.to_f64().unwrap_or(f64::NAN);
    let r2 = r2.to_f64().unwrap_or(f64::NAN);
    let r3 = r3.to_f64().unwrap_or(f64::NAN);
    if !r1.is_finite() || !r2.is_finite() || !r3.is_finite() {
        return f64::NAN;
    }

    let denom = r1 - 2.0 * r2 + r3;
    if denom == 0.0 {
        return f64::NAN;
    }

    let p = (r3 - r2) / denom;
    if !p.is_finite() || p == 0.0 {
        return f64::NAN;
    }

    let a2 = (1.0 - p) / p;
    let num = (n3 as f64 / n2 as f64).ln() + a2 * (n1 as f64 / n2 as f64).ln();
    let den = (r3 / r2).ln() + a2 * (r1 / r2).ln();
    if !den.is_finite() || den == 0.0 {
        return f64::NAN;
    }
    num / den
}

pub struct RVEstimator;

impl DistanceIDEstimator for RVEstimator {
    fn estimate_from_distances<F: Float>(distances: &[F]) -> f64 { regularly_varying_id(distances) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::{
        make_intrinsic_subspace_data, regression_test, test_zeros,
    };

    #[test]
    fn rv_estimator_regression() {
        regression_test::<RVEstimator>(5, 1000, 0, 5.05005440246909);
        regression_test::<RVEstimator>(7, 10000, 0, 6.9778378824587275);
    }

    #[test]
    fn rv_estimator_zeros() { test_zeros::<RVEstimator>(); }

    #[test]
    fn rv_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(10000, 0);
        let table = crate::TableWithDistance::with_distance(&data, crate::distance::Euclidean);
        let tree =
            crate::search::kdtree::KdTree::new(&table, crate::search::kdtree::AxisCycleSplit);

        let estimate = RVEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.456020902839396;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "rv estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
