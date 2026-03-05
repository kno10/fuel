use std::cmp::Ordering;

#[cfg(test)]
use crate::EuclideanDistance;
use crate::{DataAccess, DistanceFunction, MatrixDataAccess, VPTree};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedKnnOutlierScore {
    pub index: usize,
    pub score: f64,
}

/// Compute weighted KNN outlier scores for each point in the data set.
///
/// The score is the sum of distances to the `k` nearest neighbors (excluding
/// the point itself). Higher scores indicate stronger outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn weighted_knn_outlier_scores<T>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, impl DistanceFunction<T>>,
    k: usize,
) -> Vec<WeightedKnnOutlierScore> {
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    let mut scores = Vec::with_capacity(size);

    for idx in data.iter() {
        let score = if k_effective == 0 {
            0.0
        } else {
            tree.search_knn(&data.with_query_index(idx), (k_effective + 1).min(size))
                .into_iter()
                .filter(|neighbor| neighbor.index() != idx)
                .take(k_effective)
                .map(|neighbor| neighbor.distance())
                .sum::<f64>()
        };

        scores.push(WeightedKnnOutlierScore { index: idx, score });
    }

    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.index.cmp(&b.index))
    });

    scores
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn weighted_knn_outlier_ranks_remote_point_highest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![6.0, 6.0],
        ];

        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(42);
        let tree = VPTree::new(&data, 2, &mut rng);

        let scores = weighted_knn_outlier_scores(&tree, &data, 2);

        assert_eq!(scores.len(), points.len());
        assert_eq!(scores[0].index, 4);
        assert!(scores[0].score > scores[1].score);
    }

    #[test]
    fn weighted_knn_single_point_is_zero() {
        let points = vec![vec![1.0, 2.0]];

        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(11);
        let tree = VPTree::new(&data, 1, &mut rng);

        let scores = weighted_knn_outlier_scores(&tree, &data, 1);
        assert_eq!(
            scores,
            vec![WeightedKnnOutlierScore {
                index: 0,
                score: 0.0
            }]
        );
    }
}
