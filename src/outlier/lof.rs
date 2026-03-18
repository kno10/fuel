use num_traits::{Float, FromPrimitive};

use super::common::{OutlierScoreEntry, sort_outlier_scores};
#[cfg(test)]
use crate::distance::EuclideanDistance;
use crate::{DistanceData, KnnSearch};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LofOutlierScore<F: Float> {
    pub index: usize,
    pub score: F,
}

impl<F: Float> OutlierScoreEntry<F> for LofOutlierScore<F> {
    fn index(&self) -> usize {
        self.index
    }

    fn score(&self) -> F {
        self.score
    }
}

/// Compute Local Outlier Factor (LOF) scores.
///
/// Scores around 1.0 indicate inlier behavior; scores significantly larger than
/// 1.0 indicate outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn lof_outlier_scores<S, D, F>(
    tree: &S,
    data: D,
    k: usize,
) -> Vec<LofOutlierScore<F>>
where
    F: Float + FromPrimitive + std::iter::Sum,
    D: DistanceData<F>,
    S: KnnSearch<F, D>,
{
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    if k_effective == 0 {
        return vec![LofOutlierScore {
            index: 0,
            score: F::one(),
        }];
    }

    let mut neighborhoods: Vec<Vec<(usize, F)>> = vec![Vec::new(); size];
    let mut k_distances: Vec<F> = vec![F::zero(); size];

    for idx in data.iter() {
        let neighbors: Vec<(usize, F)> = tree
            .search_knn_by_index(&data, idx, (k_effective + 1).min(size))
            .into_iter()
            .filter(|neighbor| neighbor.index != idx)
            .take(k_effective)
            .map(|neighbor| (neighbor.index, neighbor.distance))
            .collect();

        let k_distance = neighbors
            .last()
            .map_or(F::zero(), |(_, distance)| *distance);

        neighborhoods[idx] = neighbors;
        k_distances[idx] = k_distance;
    }

    let mut local_reachability_density = vec![F::zero(); size];

    for idx in 0..size {
        let neighbors = &neighborhoods[idx];
        if neighbors.is_empty() {
            local_reachability_density[idx] = F::infinity();
            continue;
        }

        let reachability_sum = neighbors
            .iter()
            .map(|(neighbor_idx, distance)| k_distances[*neighbor_idx].max(*distance))
            .sum::<F>();

        local_reachability_density[idx] = if reachability_sum > F::zero() {
            F::from_usize(neighbors.len()).unwrap_or(F::zero()) / reachability_sum
        } else {
            F::infinity()
        };
    }

    let mut scores = Vec::with_capacity(size);

    for idx in 0..size {
        let neighbors = &neighborhoods[idx];

        let score = if neighbors.is_empty() || local_reachability_density[idx].is_infinite() {
            F::one()
        } else {
            let neighbor_lrd_sum = neighbors
                .iter()
                .map(|(neighbor_idx, _)| local_reachability_density[*neighbor_idx])
                .sum::<F>();
            neighbor_lrd_sum
                / (local_reachability_density[idx]
                    * F::from_usize(neighbors.len()).unwrap_or(F::zero()))
        };

        scores.push(LofOutlierScore { index: idx, score });
    }

    sort_outlier_scores(&mut scores);

    scores
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::TableWithDistance;
    use crate::vptree::VPTree;

    use super::*;

    #[test]
    fn lof_ranks_remote_point_highest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![6.0, 6.0],
        ];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(23);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let scores = lof_outlier_scores(&tree, &data, 2);

        assert_eq!(scores.len(), points.len());
        assert_eq!(scores[0].index, 4);
        assert!(scores[0].score > 1.0);

        assert!(
            scores
                .iter()
                .filter(|entry| entry.index != 4)
                .map(|entry| entry.score)
                .all(|value| value > 0.5 && value < 2.0)
        );
    }

    #[test]
    fn lof_matches_sklearn_reference_values() {
        let points = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![2.0, 1.0]];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(123);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let scores = lof_outlier_scores(&tree, &data, 2);

        let mut by_index = scores;
        by_index.sort_by(|a, b| {
            a.index
                .cmp(&b.index)
                .then_with(|| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
        });

        let sqrt2 = 2.0_f64.sqrt();
        let s0 = 2.0 * sqrt2 / (1.0 + sqrt2);
        let s1 = (1.0 + sqrt2) * (1.0 / (4.0 * sqrt2) + 1.0 / 2.0f64.mul_add(sqrt2, 2.0));

        assert!((by_index[0].score - s0).abs() < 1e-12);
        assert!((by_index[1].score - s1).abs() < 1e-12);
        assert!((by_index[2].score - s1).abs() < 1e-12);
    }

    #[test]
    fn lof_toy_sample_outlier_order_matches_sklearn() {
        let points = vec![
            vec![-2.0, -1.0],
            vec![-1.0, -1.0],
            vec![-1.0, -2.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![5.0, 3.0],
            vec![-4.0, 2.0],
        ];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(777);
        let tree: VPTree<f64> = VPTree::new(&data, 4, &mut rng);

        let scores = lof_outlier_scores(&tree, &data, 5);
        assert_eq!(scores.len(), points.len());

        let mut by_index = scores;
        by_index.sort_by(|a, b| {
            a.index
                .cmp(&b.index)
                .then_with(|| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
        });

        let max_inlier = by_index[..6]
            .iter()
            .map(|entry| entry.score)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_outlier = by_index[6..]
            .iter()
            .map(|entry| entry.score)
            .fold(f64::INFINITY, f64::min);

        assert!(
            max_inlier < min_outlier,
            "expected toy outliers to have higher LOF than inliers; max_inlier={max_inlier}, min_outlier={min_outlier}"
        );
    }

    #[test]
    fn lof_duplicate_heavy_sample_has_valid_scores() {
        let mut points: Vec<Vec<f64>> = Vec::new();
        points.extend((0..50).map(|_| vec![0.1]));
        points.extend((0..150).map(|i| vec![0.1 + (i as f64) * 0.001]));
        points.extend((0..10).map(|i| vec![50.0 + i as f64]));

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(20260301);
        let tree: VPTree<f64> = VPTree::new(&data, 8, &mut rng);
        let scores = lof_outlier_scores(&tree, &data, 5);

        assert!(scores.iter().all(|entry| !entry.score.is_nan()));

        let max_score = scores
            .iter()
            .map(|entry| entry.score)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_score = scores
            .iter()
            .map(|entry| entry.score)
            .fold(f64::INFINITY, f64::min);

        assert!(max_score > 1.0);
        assert!(min_score < 1.0);
    }
}
