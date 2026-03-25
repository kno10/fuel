use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

pub fn probability_weighted_moments_2(distances: &[f64]) -> f64 {
    let n = distances.len();
    let mut begin = 0;
    while begin < n && distances[begin] <= 0.0 {
        begin += 1;
    }
    let k = n - begin;
    if k <= 3 {
        if k == 2 {
            let v1 = distances[begin] / distances[begin + 1];
            return v1 / (1.0 - v1);
        }
        if k == 3 {
            let v1 = distances[begin + 1] * 0.5 / distances[begin + 2];
            return v1 / (1.0 - 2.0 * v1);
        }
        return f64::NAN;
    }

    let last = begin + k - 1;
    let mut v2 = 0.0;
    let mut valid = 0.0;

    for &d in distances[begin..last].iter() {
        valid += 1.0;
        v2 += d * (valid + 1.0) * valid;
    }

    let w = distances[last];
    if w.is_nan() || w <= 0.0 || valid <= 0.0 {
        return f64::NAN;
    }

    v2 /= (valid + 2.0) * w * (valid + 1.0) * valid;
    v2 / (1.0 - 3.0 * v2)
}

pub struct ProbabilityWeightedMoments2;
pub type PWM2Estimator = ProbabilityWeightedMoments2;

impl DistanceBasedIntrinsicDimensionalityEstimator for ProbabilityWeightedMoments2 {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        probability_weighted_moments_2(distances)
    }
}

pub fn probability_weighted_moments_2_from_knn<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize,
) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    ProbabilityWeightedMoments2::estimate_from_knn(tree, data, query_idx, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::{
        KnnBasedIntrinsicDimensionalityEstimator, make_hypersphere_embedded_data, regression_test,
        test_zeros,
    };

    #[test]
    fn probability_weighted_moments_2_function_regression() {
        let v = probability_weighted_moments_2(&[1.0, 2.0, 3.0, 4.0]);
        assert!(v.is_finite());
        assert_eq!(v, ProbabilityWeightedMoments2::estimate_from_distances(&[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn pwm2_estimator_regression() {
        regression_test::<PWM2Estimator>(5, 1000, 0, 4.88144168533192);
        regression_test::<PWM2Estimator>(7, 10000, 0, 6.9603914177038435);
    }

    #[test]
    fn pwm2_estimator_zeros() { test_zeros::<PWM2Estimator>(); }

    #[test]
    fn pwm2_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(10000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = PWM2Estimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.463922454225947;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "pwm2 estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
