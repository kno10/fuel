use crate::intrinsicdimensionality::DistanceIDEstimator;

/// Aggregated Hill intrinsic dimensionality estimator.
///
/// Based on Hill's maximum likelihood estimator for tail index, extended via
/// aggregation across the sorted distance sequence.
///
/// For ordered distances \(0 < x_1 \le x_2 \le \dots \le x_k\), standard Hill:
/// \(\hat{m}_{Hill} = \left(\frac{1}{k-1} \sum_{i=1}^{k-1} \ln \frac{x_k}{x_i}\right)^{-1}\)
///
/// Aggregated Hill uses the cumulative formulation:
/// \(\hat{m}_{agg} = -\frac{k}{\sum_{i=1}^{k-1} [\frac{1}{i} \sum_{j=1}^{i} \ln x_j - \ln x_{i+1}]}\)
///
/// Returns `NaN` if fewer than two positive finite distances exist.
pub fn aggregated_hill_id(distances: &[f64]) -> f64 {
    let begin = crate::intrinsicdimensionality::find_begin(distances);
    let n = distances.len();
    if n - begin < 2 {
        return f64::NAN;
    }

    let (mut sum, mut hsum, mut valid) = (distances[begin].ln(), 0.0, 1);

    for &v in distances[begin + 1..].iter() {
        if !v.is_finite() || v <= 0.0 {
            continue;
        }
        let logv = v.ln();
        hsum += sum / (valid as f64) - logv;
        sum += logv;
        valid += 1;
    }

    if valid < 2 || hsum == 0.0 { f64::NAN } else { -((valid as f64) / hsum) }
}

/// Type wrapper for aggregated hill ID estimator as a `DistanceIDEstimator`.
///
/// Implements `DistanceIDEstimator` by calling [`aggregated_hill_id`].
pub struct AggregatedHillID;

impl DistanceIDEstimator for AggregatedHillID {
    fn estimate_from_distances(distances: &[f64]) -> f64 { aggregated_hill_id(distances) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::{
        make_intrinsic_subspace_data, regression_test, test_zeros,
    };

    #[test]
    fn aggregated_hill_estimator_regression() {
        regression_test::<AggregatedHillID>(5, 1000, 0, 4.710215390349222);
        regression_test::<AggregatedHillID>(7, 10000, 0, 6.947193258582592);
    }

    #[test]
    fn aggregated_hill_estimator_zeros() { test_zeros::<AggregatedHillID>(); }

    #[test]
    fn aggregated_hill_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = AggregatedHillID::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.9471075219107155;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "aggregated hill estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
