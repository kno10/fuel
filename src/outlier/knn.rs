use super::common::{OutlierResult, for_each_knn, make_outlier_result};
#[cfg(test)]
use crate::distance::Euclidean;
use crate::{DistanceData, Float, KnnSearch};

/// Compute KNN-based outlier scores for each point in the data set.
///
/// The score is the distance to the k-th nearest neighbor (excluding the point itself).
/// Higher scores indicate stronger outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn k_nearest_neighbors_outlier<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    assert!(k > 0, "k must be greater than 0");

    let size = data.len();
    let k_effective = k.min(size.saturating_sub(1));

    let scores: Vec<F> = for_each_knn(tree, data, k_effective, false, |_, neighbors| {
        neighbors.last().map_or(F::zero(), |(_, distance)| *distance)
    });

    make_outlier_result(scores, "kNN Outlier Score", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;
    use crate::search::kdtree::{KdTree, MaxVarianceSplit};
    use crate::search::vptree::VPTree;

    fn sample_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], vec![0.1, 0.1], vec![6.0, 6.0]]
    }

    #[test]
    fn knn_outlier_ranks_remote_point_highest_vp() {
        let points = sample_points();

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = k_nearest_neighbors_outlier(&tree, &data, 2);

        assert_eq!(results.scores.len(), points.len());
        assert!(results.scores[4] > results.scores[0]);
    }

    #[test]
    fn knn_outlier_ranks_remote_point_highest_kd() {
        let points = sample_points();

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree = KdTree::new(&data, MaxVarianceSplit);

        let results = k_nearest_neighbors_outlier(&tree, &data, 2);

        assert_eq!(results.scores.len(), points.len());
        assert!(results.scores[4] > results.scores[0]);
    }

    #[test]
    fn knn_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = k_nearest_neighbors_outlier(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("KNN-10").expect("No reference data for KNN-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "KNN-10",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("KNN-10", &result.scores, expected, 1e-6);
    }
}
