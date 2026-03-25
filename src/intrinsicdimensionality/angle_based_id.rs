use crate::api::IndexQuery;
use crate::intrinsicdimensionality::KnnBasedIntrinsicDimensionalityEstimator;

/// RABID (angle-based) estimator requiring neighbor graph per query.
pub struct RABIDEstimator;

/// Functional API for RABID estimator.
pub fn rabid_estimate<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    RABIDEstimator::estimate_from_knn(tree, data, query_idx, k)
}

/// Functional API for ABID estimator.
pub fn abid_estimate<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    ABIDEstimator::estimate_from_knn(tree, data, query_idx, k)
}

impl RABIDEstimator {
    pub fn compute_raw_ssq<'a, S, D, F>(
        tree: &S, data: &'a D, query_idx: usize, k: usize,
    ) -> (f64, usize)
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        let query = data.query().with_index(query_idx);

        let neighbors: Vec<_> = tree
            .search_knn(&query, k + 1)
            .into_iter()
            .filter(|n| n.index != query_idx)
            .take(k)
            .collect();

        let mut k_valid = 0;
        let mut ssq = 0.0;
        let issquared = false;

        for (i, ni) in neighbors.iter().enumerate() {
            let kdi = ni.distance.to_f64().unwrap_or(f64::INFINITY);
            if kdi <= 0.0 {
                continue;
            }
            k_valid += 1;
            let di2 = if issquared { kdi } else { kdi * kdi };
            for nj in neighbors.iter().skip(i + 1) {
                let kdj = nj.distance.to_f64().unwrap_or(f64::INFINITY);
                if kdj <= 0.0 {
                    continue;
                }
                let dj2 = if issquared { kdj } else { kdj * kdj };
                let v = data.distance(ni.index, nj.index).to_f64().unwrap_or(f64::INFINITY);
                let v2 = if issquared { v } else { v * v };
                let numerator = di2 + dj2 - v2;
                let cos2 = numerator * numerator / (4.0 * di2 * dj2);
                ssq += cos2;
            }
        }

        (ssq, k_valid)
    }
}

impl KnnBasedIntrinsicDimensionalityEstimator for RABIDEstimator {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        let (ssq, k_valid) = RABIDEstimator::compute_raw_ssq(tree, data, query_idx, k);
        if k_valid == 0 || ssq == 0.0 {
            return f64::NAN;
        }
        (k_valid as f64) * (k_valid as f64) / (2.0 * ssq)
    }
}

pub struct ABIDEstimator;

impl KnnBasedIntrinsicDimensionalityEstimator for ABIDEstimator {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        let (ssq, k_valid) = RABIDEstimator::compute_raw_ssq(tree, data, query_idx, k);
        if k_valid == 0 || ssq == 0.0 {
            return f64::NAN;
        }
        (k_valid as f64) * (k_valid as f64) / (2.0 * ssq + (k_valid as f64))
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn rabid_estimator_query_graph_non_panic() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let data =
            crate::TableWithDistance::with_distance(&points, crate::distance::EuclideanDistance);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let tree: crate::vptree::VPTree<f64> = crate::vptree::VPTree::new(&data, 2, &mut rng);
        let _ = RABIDEstimator::estimate_from_knn(&tree, &data, 0, 1);
    }

    #[test]
    fn rabid_estimator_hypersphere_close_to_5() {
        let data = crate::intrinsicdimensionality::make_hypersphere_embedded_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = RABIDEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.13370702430905;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "rabid estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }

    #[test]
    fn abid_estimator_query_graph_non_panic() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let data =
            crate::TableWithDistance::with_distance(&points, crate::distance::EuclideanDistance);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let tree: crate::vptree::VPTree<f64> = crate::vptree::VPTree::new(&data, 2, &mut rng);
        let _ = ABIDEstimator::estimate_from_knn(&tree, &data, 0, 1);
    }

    #[test]
    fn abid_estimator_hypersphere_close_to_5() {
        let data = crate::intrinsicdimensionality::make_hypersphere_embedded_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = ABIDEstimator::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.880619445228746;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "abid estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
