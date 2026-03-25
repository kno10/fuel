use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

/// Hill estimator of intrinsic dimensionality (maximum likelihood for tail).
/// Uses sorted neighbor distances (excluding query point) as input.
pub struct HillEstimator;

pub fn hill_estimate_from_distances(distances: &[f64]) -> f64 {
    HillEstimator::estimate_from_distances(distances)
}

pub fn hill_estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    HillEstimator::estimate_from_knn(tree, data, query_idx, k)
}

impl DistanceBasedIntrinsicDimensionalityEstimator for HillEstimator {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        let n = distances.len();
        if n < 2 {
            return f64::NAN;
        }
        let w = distances[n - 1];
        if w.is_nan() || w <= 0.0 {
            return f64::NAN;
        }
        let halfw = 0.5 * w;
        let mut sum = 0.0;
        let mut valid = 0;
        for &v in &distances[..n - 1] {
            if v.is_nan() || v <= 0.0 {
                continue;
            }
            if v < halfw {
                sum += (v / w).ln();
            } else {
                sum += ((v - w) / w).ln_1p();
            }
            valid += 1;
        }
        if valid < 1 || sum == 0.0 {
            return f64::NAN;
        }
        -((valid as f64) / sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::{
        KnnBasedIntrinsicDimensionalityEstimator, make_hypersphere_embedded_data, regression_test,
        test_zeros,
    };

    #[test]
    fn hill_estimator_regression() {
        regression_test::<HillEstimator>(5, 1000, 0, 4.848665990083162);
        regression_test::<HillEstimator>(7, 10000, 0, 6.945428878740164);
    }

    #[test]
    fn hill_estimator_zeros() { test_zeros::<HillEstimator>(); }

    #[test]
    fn hill_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = HillEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.443823483941868;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "hill estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }

    #[test]
    fn hill_estimator_k_small() {
        let data = make_hypersphere_embedded_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(&data, crate::distance::EuclideanDistance);
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = HillEstimator::estimate_from_knn(&tree, &table, 0, 11);
        eprintln!("Hill k=11 estimate {}", estimate);
        assert!(estimate.is_finite());
    }
}
