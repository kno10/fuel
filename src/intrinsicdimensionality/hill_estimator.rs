use crate::Float;
use crate::intrinsicdimensionality::{DistanceIDEstimator, find_begin, positive_f64};

/// Hill estimator of intrinsic dimensionality (maximum likelihood for tail).
///
/// Reference:
/// - B. M. Hill, "A simple general approach to inference about the tail of a distribution", Annals of Statistics, 1975.
/// - E. Levina, P. J. Bickel, "Maximum Likelihood Estimation of Intrinsic Dimension", NIPS 2004.
///
/// For sorted distances \(x_1 \le x_2 \le \dots \le x_n\) (excluding the query point),
/// the estimator is
/// \(\hat{m} = -\frac{k}{\sum_{i=1}^{k} \ln(x_i/x_{k+1})}\), with \(k = n-1\).
///
/// This function uses numerically stable evaluation of log ratios via `ln` and `ln_1p`.
/// If there are fewer than 2 valid distances, returns `f64::NAN`.
pub struct HillID;

pub fn hill_id<F: Float>(distances: &[F]) -> f64 {
    let begin = find_begin(distances);
    let n = distances.len();
    if n - begin < 2 {
        return f64::NAN;
    }
    let w64 = positive_f64(distances[n - 1]);
    if w64.is_nan() {
        return f64::NAN;
    }
    let halfw = 0.5 * w64;

    let (mut sum, mut valid) = (0.0, 0);
    for &v in &distances[begin..n - 1] {
        let v64 = positive_f64(v);
        if v64.is_nan() {
            continue;
        }
        sum += if v64 < halfw { (v64 / w64).ln() } else { ((v64 - w64) / w64).ln_1p() };
        valid += 1;
    }

    if valid < 1 || sum == 0.0 {
        return f64::NAN;
    }

    -((valid as f64) / sum)
}

impl DistanceIDEstimator for HillID {
    fn estimate_from_distances<F: Float>(distances: &[F]) -> f64 { hill_id(distances) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::{
        make_intrinsic_subspace_data, regression_test, test_zeros,
    };

    #[test]
    fn hill_estimator_regression() {
        regression_test::<HillID>(5, 1000, 0, 4.848665990083162);
        regression_test::<HillID>(7, 10000, 0, 6.945428878740164);
    }

    #[test]
    fn hill_estimator_zeros() { test_zeros::<HillID>(); }

    #[test]
    fn hill_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = crate::TableWithDistance::with_distance(&data, crate::distance::Euclidean);
        let tree =
            crate::search::kdtree::KdTree::new(&table, crate::search::kdtree::AxisCycleSplit);

        let estimate = HillID::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.922556491645347;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "hill estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }

    #[test]
    fn hill_estimator_k_small() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = crate::TableWithDistance::with_distance(&data, crate::distance::Euclidean);
        let tree =
            crate::search::kdtree::KdTree::new(&table, crate::search::kdtree::AxisCycleSplit);

        let estimate = HillID::estimate_from_knn(&tree, &table, 0, 11);
        eprintln!("Hill k=11 estimate {}", estimate);
        assert!(estimate.is_finite());
    }
}
