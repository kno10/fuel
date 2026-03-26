#[cfg(test)]
use crate::distance::Euclidean;
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch};

/// Compute weighted KNN outlier scores for each point in the data set.
///
/// The score is the sum of distances to the `k` nearest neighbors (excluding
/// the point itself). Higher scores indicate stronger outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn weighted_knn<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
where
    F: Float + std::iter::Sum + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    assert!(k > 0, "k must be greater than 0");

    let k_effective = k.min(data.size().saturating_sub(1));

    let scores: Vec<F> =
        crate::outlier::common::for_each_knn(tree, data, k_effective, false, |_idx, neighbors| {
            neighbors.iter().map(|(_, distance)| *distance).sum::<F>()
        });

    make_outlier_result(scores, "WeightedKNN", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;
    use crate::vptree::VPTree;

    #[test]
    fn weighted_knn_outlier_ranks_remote_point_highest() {
        let points =
            vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], vec![6.0, 6.0], vec![0.1, 0.1]];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = weighted_knn(&tree, &data, 2);

        assert_eq!(results.scores.len(), points.len());
        let (best_index, best_score) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
        assert!(*best_score > 0.0);
    }

    #[test]
    fn knnw_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = weighted_knn(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("KNNW-10").expect("No reference for KNNW-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "KNNW-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("KNNW-10", &result.scores, expected, 1e-6);
    }
}
