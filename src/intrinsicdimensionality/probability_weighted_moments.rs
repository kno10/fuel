use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

pub fn probability_weighted_moments(distances: &[f64]) -> f64 {
    let n = distances.len();
    let mut begin = 0;
    while begin < n && distances[begin] <= 0.0 {
        begin += 1;
    }
    let k = n - begin;
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
        valid += 1.0;
        v1 += d * valid;
    }

    let w = distances[last];
    if w.is_nan() || w <= 0.0 || valid <= 0.0 {
        return f64::NAN;
    }

    v1 /= (valid + 1.0) * w * valid;
    v1 / (1.0 - 2.0 * v1)
}

pub struct ProbabilityWeightedMoments;
pub type PWMEstimator = ProbabilityWeightedMoments;

impl DistanceBasedIntrinsicDimensionalityEstimator for ProbabilityWeightedMoments {
    fn estimate_from_distances(distances: &[f64]) -> f64 { probability_weighted_moments(distances) }
}

pub fn probability_weighted_moments_from_knn<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize,
) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    ProbabilityWeightedMoments::estimate_from_knn(tree, data, query_idx, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::{
        KnnBasedIntrinsicDimensionalityEstimator, make_hypersphere_embedded_data, regression_test,
        test_zeros,
    };

    #[test]
    fn probability_weighted_moments_function_regression() {
        let v = probability_weighted_moments(&[1.0, 2.0, 3.0, 4.0]);
        assert!(v.is_finite());
        assert_eq!(v, ProbabilityWeightedMoments::estimate_from_distances(&[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn pwm_estimator_regression() {
        regression_test::<PWMEstimator>(5, 1000, 0, 4.891606982612564);
        regression_test::<PWMEstimator>(7, 10000, 0, 6.959066158235904);
    }

    #[test]
    fn pwm_estimator_zeros() { test_zeros::<PWMEstimator>(); }

    #[test]
    fn pwm_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(10000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = PWMEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.703023761602155;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "pwm estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
