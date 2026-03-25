use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

pub fn rv_estimate_from_distances(distances: &[f64]) -> f64 {
    RVEstimator::estimate_from_distances(distances)
}

pub fn rv_estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    RVEstimator::estimate_from_knn(tree, data, query_idx, k)
}

pub struct RVEstimator;

impl DistanceBasedIntrinsicDimensionalityEstimator for RVEstimator {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        let n = distances.len();
        let mut begin = 0;
        while begin < n && distances[begin] <= 0.0 {
            begin += 1;
        }
        let k = n - begin;
        if k < 2 {
            return f64::NAN;
        }
        let n1 = k >> 1;
        let n2 = (3 * k) >> 2;
        let n3 = k;
        if n1 == 0 || n2 == 0 || n3 == 0 || begin + n3 > n {
            return f64::NAN;
        }

        let r1 = distances[begin + n1 - 1];
        let r2 = distances[begin + n2 - 1];
        let r3 = distances[begin + n3 - 1];

        if r1.is_nan() || r2.is_nan() || r3.is_nan() || r1 <= 0.0 || r2 <= 0.0 || r3 <= 0.0 {
            return f64::NAN;
        }
        let denom = r1 - 2.0 * r2 + r3;
        if denom == 0.0 {
            return f64::NAN;
        }

        let p = (r3 - r2) / denom;
        if !p.is_finite() || p == 0.0 {
            return f64::NAN;
        }

        let a2 = (1.0 - p) / p;
        let num = (n3 as f64 / n2 as f64).ln() + a2 * (n1 as f64 / n2 as f64).ln();
        let den = (r3 / r2).ln() + a2 * (r1 / r2).ln();
        if !den.is_finite() || den == 0.0 {
            return f64::NAN;
        }
        num / den
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
    fn rv_estimator_regression() {
        regression_test::<RVEstimator>(5, 1000, 0, 5.05005440246909);
        regression_test::<RVEstimator>(7, 10000, 0, 6.9778378824587275);
    }

    #[test]
    fn rv_estimator_zeros() { test_zeros::<RVEstimator>(); }

    #[test]
    fn rv_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(10000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = RVEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.38988116527122;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "rv estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
