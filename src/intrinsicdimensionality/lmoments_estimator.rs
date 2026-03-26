use crate::intrinsicdimensionality::DistanceIDEstimator;
use crate::intrinsicdimensionality::method_of_moments::method_of_moments_id;
use crate::statistics::probability_weighted_moments::sam_lmr;

/// L-moments estimator of intrinsic dimensionality.
///
/// Reference:
///
/// L. Amsaleg, O. Chelly, T. Furon, S. Girard, M. E. Houle, K. Kawarabayashi, M. Nett
/// Estimating Local Intrinsic Dimensionality
/// Proc. SIGKDD International Conference on Knowledge Discovery and Data Mining 2015
///
/// J. R. M. Hosking
/// Fortran routines for use with the method of L-moments Version 3.03
/// IBM Research Technical Report
///
/// Uses sample L-moments \(\lambda_1, \lambda_2\) from sorted distances.
/// \\(\hat{m} = -\frac{1}{2w} \left(\frac{\lambda_1^2}{\lambda_2} - \lambda_1\right)\\)
/// where \(w\) is the maximum distance. Falls back to method-of-moments when only 2 points.
///
/// For stability, if \(\lambda_2=0\), uses first L-moment fallback.
pub struct LMomentsEstimator;

pub fn lmoments_id(distances: &[f64]) -> f64 {
    LMomentsEstimator::estimate_from_distances(distances)
}

impl DistanceIDEstimator for LMomentsEstimator {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        let n = distances.len();
        let mut begin = 0;
        while begin < n && distances[begin] <= 0.0 {
            begin += 1;
        }
        let len = n - begin;
        if len < 2 {
            return f64::NAN;
        }

        if len == 2 {
            return method_of_moments_id(&distances[begin..]);
        }

        let w = distances[n - 1];
        if w.is_nan() || w <= 0.0 {
            return f64::NAN;
        }

        let lmom = sam_lmr(distances[begin..].iter().copied().rev(), 2);
        if lmom.len() < 2 || lmom[1] == 0.0 {
            // fallback to first moment only
            let l1 = lmom.first().copied().unwrap_or(0.0);
            return -0.5 * (l1 * 2.0) / w * ((len as f64) + 0.5) * (len as f64);
        }

        let (l1, l2) = (lmom[0], lmom[1]);
        -0.5 * ((l1 * l1 / l2) - l1) / w
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
    fn lmoments_estimator_regression() {
        regression_test::<LMomentsEstimator>(5, 1000, 0, 4.882011622061694);
        regression_test::<LMomentsEstimator>(7, 10000, 0, 6.96093691219275);
    }

    #[test]
    fn lmoments_estimator_zeros() { test_zeros::<LMomentsEstimator>(); }

    #[test]
    fn lmoments_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table =
            crate::data::TableWithDistance::with_distance(&data, crate::distance::Euclidean);
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = LMomentsEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.138352302606048;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "lmoments estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
