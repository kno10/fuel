use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

pub fn method_of_moments(distances: &[f64]) -> f64 {
    let n = distances.len();
    if n < 2 {
        return f64::NAN;
    }
    let last = n - 1;
    let mut v1 = 0.0;
    let mut valid = 0;
    for &v in &distances[..last] {
        if v > 0.0 {
            v1 += v;
            valid += 1;
        }
    }
    if valid <= 1 {
        return f64::NAN;
    }
    let w = distances[last];
    v1 /= (valid as f64) * w;
    if v1 >= 1.0 {
        return f64::INFINITY;
    }
    v1 / (1.0 - v1)
}

pub struct MethodOfMoments;
pub type MOMEstimator = MethodOfMoments;

impl DistanceBasedIntrinsicDimensionalityEstimator for MethodOfMoments {
    fn estimate_from_distances(distances: &[f64]) -> f64 { method_of_moments(distances) }
}

pub fn method_of_moments_from_knn<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize,
) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    MethodOfMoments::estimate_from_knn(tree, data, query_idx, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::{
        KnnBasedIntrinsicDimensionalityEstimator, make_hypersphere_embedded_data, regression_test,
        test_zeros,
    };

    #[test]
    fn method_of_moments_regression() {
        let v = method_of_moments(&[1.0, 2.0, 3.0, 4.0]);
        assert!(v.is_finite());
        assert_eq!(v, MethodOfMoments::estimate_from_distances(&[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn mom_estimator_regression() {
        regression_test::<MOMEstimator>(5, 1000, 0, 4.8704752769340836);
        regression_test::<MOMEstimator>(7, 10000, 0, 6.946161496762817);
    }

    #[test]
    fn mom_estimator_zeros() { test_zeros::<MOMEstimator>(); }

    #[test]
    fn mom_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(10000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = MOMEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.375015257293149;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "mom estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
