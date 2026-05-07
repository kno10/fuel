use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

/// Local Isolation Coefficient (LIC).
///
/// score = kNN-distance + average distance to k nearest neighbors
///
/// Reference: B. Yu, M. Song, L. Wang (2009).
pub fn local_isolation_coefficient<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(vec![], "LIC", false, F::zero(), F::zero(), F::infinity());
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::zero(); size],
            "LIC",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods: Vec<Vec<(usize, F)>> =
        for_each_knn(tree, data, k_effective, false, |_idx, neighbors| neighbors);

    let scores: Vec<F> = (0..size)
        .par_map(|i| {
            let neigh = &neighborhoods[i];
            let knn_distance = neigh.last().map(|(_, d)| *d).unwrap_or(F::zero());
            let sum: F = neigh.iter().map(|(_, d)| *d).sum();
            let n = F::from_usize(neigh.len()).unwrap_or(F::zero());
            let mean = if n > F::zero() { sum / n } else { F::zero() };
            knn_distance + mean
        });

    make_outlier_result(scores, "LIC", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn lic_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = local_isolation_coefficient(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("LIC-10").expect("No reference for LIC-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "LIC-10",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("LIC-10", &result.scores, expected, 1e-6);
    }
}
