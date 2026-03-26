use crate::intrinsicdimensionality::DistanceIDEstimator;

/// Probability weighted moments intrinsic dimensionality estimator.
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
/// We use the unbiased weights and treat the first entry as if having an extra 0 point.
///
/// Computes weighted means from ordered distances and solves
/// \(v_2 = \frac{\sum_{i=1}^{k-1} i d_i}{(k+1) k w}\)
/// then ID \(\hat{m} = \frac{v_2}{1-2v_2}\).
///
/// Falls back to analytic ratio when only 2 distances are available.
pub fn probability_weighted_moments_id(distances: &[f64]) -> f64 {
    let begin = crate::intrinsicdimensionality::find_begin(distances);

    let k = distances.len() - begin;
    if k < 2 {
        return f64::NAN;
    }

    if k == 2 {
        let v1 = distances[begin] / distances[begin + 1];
        return v1 / (1.0 - v1);
    }

    let last = begin + k - 1;
    let mut v1 = 0.0;
    let mut valid = 0.0;
    for &d in &distances[begin..last] {
        if !d.is_finite() || d <= 0.0 {
            continue;
        }
        valid += 1.0;
        v1 += d * valid;
    }

    let w = distances[last];
    if !w.is_finite() || w <= 0.0 || valid <= 0.0 {
        return f64::NAN;
    }

    let v2 = v1 / ((valid + 1.0) * w * valid);
    v2 / (1.0 - 2.0 * v2)
}

pub struct ProbabilityWeightedMoments;

impl DistanceIDEstimator for ProbabilityWeightedMoments {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        probability_weighted_moments_id(distances)
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
    fn probability_weighted_moments_function_regression() {
        let v = probability_weighted_moments_id(&[1.0, 2.0, 3.0, 4.0]);
        assert!(v.is_finite());
        assert_eq!(v, ProbabilityWeightedMoments::estimate_from_distances(&[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn pwm_estimator_regression() {
        regression_test::<ProbabilityWeightedMoments>(5, 1000, 0, 4.891606982612564);
        regression_test::<ProbabilityWeightedMoments>(7, 10000, 0, 6.959066158235904);
    }

    #[test]
    fn pwm_estimator_zeros() { test_zeros::<ProbabilityWeightedMoments>(); }

    #[test]
    fn pwm_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(10000, 0);
        let table =
            crate::data::TableWithDistance::with_distance(&data, crate::distance::Euclidean);
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = ProbabilityWeightedMoments::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.4819050834308065;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "pwm estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
