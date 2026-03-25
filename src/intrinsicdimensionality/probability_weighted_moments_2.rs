use crate::intrinsicdimensionality::DistanceIDEstimator;

/// Second variant of probability weighted moments intrinsic dimensionality estimator.
///
/// Reference:
///
/// L. Amsaleg, O. Chelly, T. Furon, S. Girard, M. E. Houle, K. Kawarabayashi, M. Nett
/// Estimating Local Intrinsic Dimensionality
/// Proc. SIGKDD International Conference on Knowledge Discovery and Data Mining 2015
///
/// J. Maciunas Landwehr, N. C. Matalas, J. R. Wallis
/// Probability weighted moments compared with some traditional techniques in estimating Gumbel parameters and quantiles
/// Water Resources Research 15(5)
///
/// It uses a second PWM and theoretically has higher variance. Included for completeness.
///
/// For small k, uses analytic closed forms:
/// - k=2: \(v_1 = d_1 / d_2\), \(\hat{m} = v_1/(1-v_1)\)
/// - k=3: \(v_1 = d_2 / (2 d_3)\), \(\hat{m} = v_1/(1-2v_1)\)
///
/// For general k, weighted moment ratio is:
/// \(v_2 = \frac{\sum_{i=1}^{k-1} i(i+1) d_i}{(k+2)(k+1)k w}\)
/// and \(\hat{m} = v_2/(1-3v_2)\).
///
/// Returns `NaN` for insufficient or invalid data.
pub fn probability_weighted_moments_2_id(distances: &[f64]) -> f64 {
    let begin = crate::intrinsicdimensionality::find_begin(distances);

    let k = distances.len() - begin;
    if k < 2 {
        return f64::NAN;
    }
    if k == 2 {
        let v1 = distances[begin] / distances[begin + 1];
        return v1 / (1.0 - v1);
    }
    if k == 3 {
        let v1 = distances[begin + 1] * 0.5 / distances[begin + 2];
        return v1 / (1.0 - 2.0 * v1);
    }

    let last = begin + k - 1;
    let mut v2 = 0.0;
    let mut valid = 0.0;
    for d in distances[begin..last].iter().copied() {
        if !d.is_finite() || d <= 0.0 {
            continue;
        }
        valid += 1.0;
        v2 += d * (valid + 1.0) * valid;
    }

    let w = distances[last];
    if !w.is_finite() || w <= 0.0 || valid <= 0.0 {
        return f64::NAN;
    }

    let v2 = v2 / ((valid + 2.0) * w * (valid + 1.0) * valid);
    v2 / (1.0 - 3.0 * v2)
}

pub struct ProbabilityWeightedMoments2;

impl DistanceIDEstimator for ProbabilityWeightedMoments2 {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        probability_weighted_moments_2_id(distances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::{
        make_intrinsic_subspace_data, regression_test, test_zeros,
    };

    #[test]
    fn probability_weighted_moments_2_function_regression() {
        let v = probability_weighted_moments_2_id(&[1.0, 2.0, 3.0, 4.0]);
        assert!(v.is_finite());
        assert_eq!(v, ProbabilityWeightedMoments2::estimate_from_distances(&[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn pwm2_estimator_regression() {
        regression_test::<ProbabilityWeightedMoments2>(5, 1000, 0, 4.88144168533192);
        regression_test::<ProbabilityWeightedMoments2>(7, 10000, 0, 6.9603914177038435);
    }

    #[test]
    fn pwm2_estimator_zeros() { test_zeros::<ProbabilityWeightedMoments2>(); }

    #[test]
    fn pwm2_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(10000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = ProbabilityWeightedMoments2::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.669889575177945;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "pwm2 estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
