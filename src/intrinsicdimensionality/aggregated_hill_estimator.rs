use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

pub fn aggregated_hill_estimate_from_distances(distances: &[f64]) -> f64 {
    AggregatedHillEstimator::estimate_from_distances(distances)
}

pub fn aggregated_hill_estimate_from_knn<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize,
) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    AggregatedHillEstimator::estimate_from_knn(tree, data, query_idx, k)
}

pub struct AggregatedHillEstimator;

impl DistanceBasedIntrinsicDimensionalityEstimator for AggregatedHillEstimator {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        let n = distances.len();
        if n < 2 {
            return f64::NAN;
        }
        let mut hsum = 0.0;
        let mut sum = 0.0;
        let mut i = 0;
        let mut valid = 0;

        while i < n {
            let v = distances[i];
            i += 1;
            if v > 0.0 {
                sum = v.ln();
                valid += 1;
                break;
            }
        }

        while i < n {
            let v = distances[i];
            i += 1;
            if v.is_nan() || v <= 0.0 {
                continue;
            }
            let logv = v.ln();
            hsum += sum / (valid as f64) - logv;
            valid += 1;
            sum += logv;
        }

        if valid < 1 || hsum == 0.0 {
            return f64::NAN;
        }

        -((valid as f64) / hsum)
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
    fn aggregated_hill_estimator_regression() {
        regression_test::<AggregatedHillEstimator>(5, 1000, 0, 4.710215390349222);
        regression_test::<AggregatedHillEstimator>(7, 10000, 0, 6.947193258582592);
    }

    #[test]
    fn aggregated_hill_estimator_zeros() { test_zeros::<AggregatedHillEstimator>(); }

    #[test]
    fn aggregated_hill_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = AggregatedHillEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.735992849774032;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "aggregated hill estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
