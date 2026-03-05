use std::cmp::Ordering;

#[cfg(test)]
use crate::EuclideanDistance;
use crate::{DataAccess, DistanceFunction, MatrixDataAccess, VPTree};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnnOutlierScore {
    pub index: usize,
    pub score: f64,
}

/// Compute KNN-based outlier scores for each point in the data set.
///
/// The score is the distance to the k-th nearest neighbor (excluding the point itself).
/// Higher scores indicate stronger outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn knn_outlier_scores<T>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, impl DistanceFunction<T>>,
    k: usize,
) -> Vec<KnnOutlierScore> {
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    let mut scores = Vec::with_capacity(size);

    for idx in data.iter() {
        let score = if k_effective == 0 {
            0.0
        } else {
            let neighbors =
                tree.search_knn(&data.with_query_index(idx), (k_effective + 1).min(size));
            let rank = k_effective - 1;
            neighbors
                .into_iter()
                .filter(|neighbor| neighbor.index() != idx)
                .nth(rank)
                .map_or(0.0, |neighbor| neighbor.distance())
        };

        scores.push(KnnOutlierScore { index: idx, score });
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
    fn knn_outlier_ranks_remote_point_highest() {
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

        let scores = knn_outlier_scores(&tree, &data, 2);

        assert_eq!(scores.len(), points.len());
        assert_eq!(scores[0].index, 4);
        assert!(scores[0].score > scores[1].score);
    }
}
