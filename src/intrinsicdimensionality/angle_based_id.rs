use crate::api::IndexQuery;
use crate::intrinsicdimensionality::KNNIDEstimator;

/// Angle-based ID estimators (ABID/RABID).
///
/// References:
/// - Erik Thordsen and Erich Schubert, "ABID: Angle Based Intrinsic Dimensionality",
///   Proc. 13th Int. Conf. Similarity Search and Applications (SISAP'2020).
///
/// RABID is the raw angle-based estimator; ABID adds a small bias correction.
///
/// For neighbor distances and mutual angles, the ABID/RABID estimators approximate the
/// intrinsic dimension by the relationship:
/// \( E[\cos^2(\theta)] \approx 1 / (2m) \) for dimension \(m\).
///
/// The final ID estimators are:
/// - RABID: \( \hat{m} = \frac{k^2}{2 \sum \cos^2(\theta)} \)
/// - ABID: \( \hat{m} = \frac{k^2}{2 \sum \cos^2(\theta) + k} \)
///
/// Functional API for RABID estimator.
pub fn rabid<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    RABID::estimate_from_knn(tree, data, query_idx, k)
}

/// Functional API for ABID estimator.
pub fn abid<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: crate::Float,
    D: crate::DistanceData<F> + 'a,
    S: crate::KnnSearch<F, D::Query<'a>> + Sync,
{
    ABID::estimate_from_knn(tree, data, query_idx, k)
}

/// RABID estimator type.
///
/// Computes intrinsic dimension estimate from the average squared cosine of angles
/// between nearest-neighbor vectors.
/// See formula in module-level docs above.
pub struct RABID;

// TODO: add an API to DistanceData to get is_squared from the distance.
impl RABID {
    pub fn compute_raw_ssq<'a, S, D, F>(
        tree: &S, data: &'a D, query_idx: usize, k: usize, is_squared: bool,
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

        for (i, ni) in neighbors.iter().enumerate() {
            let kdi = ni.distance.to_f64().unwrap_or(f64::INFINITY);
            if kdi.is_nan() || kdi <= 0.0 {
                continue;
            }
            k_valid += 1;
            let di2 = if is_squared { kdi } else { kdi * kdi };

            for nj in neighbors.iter().skip(i + 1) {
                let kdj = nj.distance.to_f64().unwrap_or(f64::INFINITY);
                if kdj.is_nan() || kdj <= 0.0 {
                    continue;
                }
                let dj2 = if is_squared { kdj } else { kdj * kdj };
                let v = data.distance(ni.index, nj.index).to_f64().unwrap_or(f64::INFINITY);
                if v.is_nan() || v <= 0.0 {
                    continue;
                }
                let v2 = if is_squared { v } else { v * v };
                let numerator = di2 + dj2 - v2;
                let cos2 = numerator * numerator / (4.0 * di2 * dj2);
                ssq += cos2;
            }
        }

        (ssq, k_valid)
    }
}

impl KNNIDEstimator for RABID {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        let (ssq, k_valid) = RABID::compute_raw_ssq(tree, data, query_idx, k, false);
        if k_valid == 0 || ssq == 0.0 {
            return f64::NAN;
        }
        (k_valid as f64) * (k_valid as f64) / (2.0 * ssq)
    }
}

/// ABID estimator type.
///
/// Adjusted RABID formula with an extra term in the denominator:
/// \(\hat{m}_{ABID} = \frac{k^2}{2S + k}\), with
/// \(S = \sum_{i<j} \cos^2(\theta_{ij})\).
pub struct ABID;

impl KNNIDEstimator for ABID {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        let (ssq, k_valid) = RABID::compute_raw_ssq(tree, data, query_idx, k, false);
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
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::intrinsicdimensionality::test::make_intrinsic_subspace_data;
    use crate::search::kdtree::{AxisCycleSplit, KdTree};
    use crate::search::vptree::VPTree;

    #[test]
    fn rabid_estimator_query_graph_non_panic() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let data = crate::TableWithDistance::with_distance(&points, crate::distance::Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);
        let _ = RABID::estimate_from_knn(&tree, &data, 0, 1);
    }

    #[test]
    fn rabid_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = TableWithDistance::with_distance(&data, Euclidean);
        let tree = KdTree::new(&table, AxisCycleSplit);

        let estimate = RABID::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 5.13903234155527;
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
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);
        let _ = ABID::estimate_from_knn(&tree, &data, 0, 1);
    }

    #[test]
    fn abid_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = TableWithDistance::with_distance(&data, Euclidean);
        let tree = KdTree::new(&table, AxisCycleSplit);

        let estimate = ABID::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.8854323914334685;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "abid estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
