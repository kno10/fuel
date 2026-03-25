use crate::intrinsicdimensionality::method_of_moments::method_of_moments;
use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};
use crate::statistics::probability_weighted_moments::sam_lmr;

pub struct LMomentsEstimator;

pub fn lmoments_estimate_from_knn<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize,
) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    LMomentsEstimator::estimate_from_knn(tree, data, query_idx, k)
}

impl DistanceBasedIntrinsicDimensionalityEstimator for LMomentsEstimator {
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
            return method_of_moments(&distances[begin..]);
        }

        let w = distances[n - 1];
        if w.is_nan() || w <= 0.0 {
            return f64::NAN;
        }

        let rev: Vec<f64> = distances[begin..].iter().rev().cloned().collect();

        let lmom = sam_lmr(&rev, 2);
        if lmom.len() < 2 || lmom[1] == 0.0 {
            // fallback to first moment only
            let l1 = lmom.first().copied().unwrap_or(0.0);
            return -0.5 * (l1 * 2.0) / w * ((len as f64) + 0.5) * (len as f64);
        }

        let l1 = lmom[0];
        let l2 = lmom[1];
        -0.5 * ((l1 * l1 / l2) - l1) / w
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
    fn lmoments_estimator_regression() {
        regression_test::<LMomentsEstimator>(5, 1000, 0, 4.8820116220616935);
        regression_test::<LMomentsEstimator>(7, 10000, 0, 6.96093691219275);
    }

    #[test]
    fn lmoments_estimator_zeros() { test_zeros::<LMomentsEstimator>(); }

    #[test]
    fn lmoments_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = LMomentsEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.54007891599851;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "lmoments estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
