use num_traits::Float;

use super::common::{OutlierScoreEntry, sort_outlier_scores};
#[cfg(test)]
use crate::distance::EuclideanDistance;
use crate::{DistanceData, KnnSearch};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedKnnOutlierScore<F: Float> {
    pub index: usize,
    pub score: F,
}

impl<F: Float> OutlierScoreEntry<F> for WeightedKnnOutlierScore<F> {
    fn index(&self) -> usize {
        self.index
    }

    fn score(&self) -> F {
        self.score
    }
}

/// Compute weighted KNN outlier scores for each point in the data set.
///
/// The score is the sum of distances to the `k` nearest neighbors (excluding
/// the point itself). Higher scores indicate stronger outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn weighted_knn_outlier_scores<S, D, F>(
    tree: &S,
    data: D,
    k: usize,
) -> Vec<WeightedKnnOutlierScore<F>>
where
    F: Float + std::iter::Sum,
    D: DistanceData<F>,
    S: KnnSearch<F, D>,
{
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    let mut scores = Vec::with_capacity(size);

    for idx in data.iter() {
        let score = if k_effective == 0 {
            F::zero()
        } else {
            tree.search_knn_by_index(&data, idx, (k_effective + 1).min(size))
                .into_iter()
                .filter(|neighbor| neighbor.index != idx)
                .take(k_effective)
                .map(|neighbor| neighbor.distance)
                .sum::<F>()
        };

        scores.push(WeightedKnnOutlierScore { index: idx, score });
    }

    sort_outlier_scores(&mut scores);

    scores
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::TableWithDistance;
    use crate::vptree::VPTree;

    use super::*;

    #[test]
    fn weighted_knn_outlier_ranks_remote_point_highest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![6.0, 6.0],
            vec![0.1, 0.1],
        ];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let scores = weighted_knn_outlier_scores(&tree, &data, 2);

        assert_eq!(scores.len(), points.len());
        assert_eq!(scores[0].index, 3);
        assert!(scores[0].score > scores[1].score);
    }
}
