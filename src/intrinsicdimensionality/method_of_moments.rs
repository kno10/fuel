use crate::Float;
use crate::intrinsicdimensionality::{DistanceIDEstimator, positive_f64};

/// Method-of-moments intrinsic dimensionality estimator.
///
/// Reference:
/// - A. Kleindessner et al., "Geometric Moment-based Estimation of Intrinsic Dimension".
/// - EM algorithms from ELKI ID estimators.
///
/// For sorted distances \(d_1, ..., d_n\), it uses critical ratio:
/// \(v_1 = \frac{1}{n-1} \sum_{i=1}^{n-1} \frac{d_i}{d_n} \) and
/// \(\hat{m} = \frac{v_1}{1-v_1} \).
///
/// Returns `NaN` for invalid / insufficient data.
pub fn method_of_moments_id<F: Float>(distances: &[F]) -> f64 {
    let len = distances.len();
    if len < 2 {
        return f64::NAN;
    }

    let w = positive_f64(distances[len - 1]);
    if w.is_nan() {
        return f64::NAN;
    }

    let (mut v1, mut valid) = (0.0, 0);
    for &d in &distances[..len - 1] {
        let d64 = positive_f64(d);
        if d64.is_nan() {
            continue;
        }
        v1 += d64;
        valid += 1;
    }

    if valid <= 1 {
        return f64::NAN;
    }

    let v1 = v1 / ((valid as f64) * w);
    if v1 >= 1.0 { f64::INFINITY } else { v1 / (1.0 - v1) }
}

pub struct MethodOfMoments;

impl DistanceIDEstimator for MethodOfMoments {
    fn estimate_from_distances<F: Float>(distances: &[F]) -> f64 { method_of_moments_id(distances) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::{
        make_intrinsic_subspace_data, regression_test, test_zeros,
    };
    use crate::search::kdtree::{AxisCycleSplit, KdTree};

    #[test]
    fn method_of_moments_regression() {
        let v = method_of_moments_id(&[1.0, 2.0, 3.0, 4.0]);
        assert!(v.is_finite());
        assert_eq!(v, MethodOfMoments::estimate_from_distances(&[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn mom_estimator_regression() {
        regression_test::<MethodOfMoments>(5, 1000, 0, 4.8704752769340836);
        regression_test::<MethodOfMoments>(7, 10000, 0, 6.946161496762817);
    }

    #[test]
    fn mom_estimator_zeros() { test_zeros::<MethodOfMoments>(); }

    #[test]
    fn mom_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(10000, 0);
        let table = TableWithDistance::with_distance(&data, Euclidean);
        let tree = KdTree::new(&table, AxisCycleSplit);

        let estimate = MethodOfMoments::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.285970290371168;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "MoM estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
