use crate::intrinsicdimensionality::{
    DistanceBasedIntrinsicDimensionalityEstimator, KnnBasedIntrinsicDimensionalityEstimator,
};

pub struct ZipfEstimator;

pub fn zipf_estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    ZipfEstimator::estimate_from_knn(tree, data, query_idx, k)
}

impl DistanceBasedIntrinsicDimensionalityEstimator for ZipfEstimator {
    fn estimate_from_distances(distances: &[f64]) -> f64 {
        let n = distances.len();
        let mut begin = 0;
        while begin < n && distances[begin] <= 0.0 {
            begin += 1;
        }
        let len = (n - begin) as f64;
        if len < 2.0 {
            return f64::NAN;
        }
        let bias = 0.6;
        let nplus1 = len + bias;

        let (mut wls, mut ws, mut ls, mut wws) = (0.0, 0.0, 0.0, 0.0);
        for (i, &v) in distances[begin..].iter().enumerate() {
            if v <= 0.0 {
                continue;
            }
            let logv = v.ln();
            let weight = (nplus1 / ((i as f64) + bias)).ln();
            wls += weight * logv;
            ws += weight;
            ls += logv;
            wws += weight * weight;
        }

        let denom = len * wws - ws * ws;
        if denom == 0.0 {
            return f64::NAN;
        }
        -1.0 / ((len * wls - ws * ls) / denom)
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
    fn zipf_estimator_regression() {
        regression_test::<ZipfEstimator>(5, 1000, 0, 4.702443328729227);
        regression_test::<ZipfEstimator>(7, 10000, 0, 6.943453727205677);
    }

    #[test]
    fn zipf_estimator_zeros() { test_zeros::<ZipfEstimator>(); }

    #[test]
    fn zipf_estimator_hypersphere_close_to_5() {
        let data = make_hypersphere_embedded_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = ZipfEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.6440800430869675;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "zipf estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
