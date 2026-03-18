use num_traits::{Float, FromPrimitive, ToPrimitive};

use super::common::{OutlierScoreEntry, sort_outlier_scores};
#[cfg(test)]
use crate::distance::EuclideanDistance;
use crate::{DistanceData, KnnSearch};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopOutlierScore<F: Float> {
    pub index: usize,
    pub score: F,
}

impl<F: Float> OutlierScoreEntry<F> for LoopOutlierScore<F> {
    fn index(&self) -> usize {
        self.index
    }

    fn score(&self) -> F {
        self.score
    }
}

/// Compute Local Outlier Probabilities (LoOP) scores.
///
/// Scores are in the range [0, 1], where larger values indicate higher
/// outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn loop_outlier_scores<S, D, F>(
    tree: &S,
    data: &D,
    k: usize,
    n_lambda: f64,
) -> Vec<LoopOutlierScore<F>>
where
    F: Float + FromPrimitive + ToPrimitive + std::iter::Sum,
    D: DistanceData<F>,
    S: KnnSearch<F, D>,
{
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    if k_effective == 0 {
        return vec![LoopOutlierScore {
            index: 0,
            score: F::zero(),
        }];
    }

    let mut neighborhoods: Vec<Vec<(usize, F)>> = vec![Vec::new(); size];

    for idx in data.iter() {
        let neighbors: Vec<(usize, F)> = tree
            .search_knn_by_index(&data, idx, (k_effective + 1).min(size))
            .into_iter()
            .filter(|neighbor| neighbor.index != idx)
            .take(k_effective)
            .map(|neighbor| (neighbor.index, neighbor.distance))
            .collect();

        neighborhoods[idx] = neighbors;
    }

    let pdists: Vec<F> = neighborhoods
        .iter()
        .map(|neighbors| {
            if neighbors.is_empty() {
                F::zero()
            } else {
                (neighbors
                    .iter()
                    .map(|(_, distance)| *distance * *distance)
                    .sum::<F>()
                    / F::from_usize(neighbors.len()).unwrap_or(F::zero()))
                .sqrt()
            }
        })
        .collect();

    let plofs: Vec<f64> = neighborhoods
        .iter()
        .enumerate()
        .map(|(idx, neighbors)| {
            if neighbors.is_empty() {
                return 0.0;
            }
            let neighbor_pdists_mean = neighbors
                .iter()
                .map(|(neighbor_idx, _)| pdists[*neighbor_idx])
                .sum::<F>()
                / F::from_usize(neighbors.len()).unwrap_or(F::zero());

            if neighbor_pdists_mean > F::zero() {
                ((pdists[idx] / neighbor_pdists_mean).to_f64().unwrap_or(1.0) - 1.0).max(0.0)
            } else {
                0.0
            }
        })
        .collect();

    let mut nplof = n_lambda
        * (plofs.iter().map(|value| value * value).sum::<f64>() / plofs.len() as f64).sqrt();

    if nplof <= 0.0 {
        nplof = 1.0;
    }

    let sqrt_2 = 2.0.sqrt();

    let mut scores: Vec<LoopOutlierScore<F>> = plofs
        .iter()
        .enumerate()
        .map(|(idx, plof)| LoopOutlierScore {
            index: idx,
            score: F::from_f64(erf_approx((plof / (nplof * sqrt_2)).max(0.0)).max(0.0))
                .unwrap_or(F::zero()),
        })
        .collect();

    sort_outlier_scores(&mut scores);

    scores
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();

    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-(abs_x * abs_x)).exp());

    sign * y
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
    fn loop_ranks_remote_point_highest() {
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

        let scores = loop_outlier_scores(&tree, &data, 2, 2.0);

        assert_eq!(scores.len(), points.len());
        assert_eq!(scores[0].index, 4);
        assert!(scores[0].score > 0.5);
        assert!(scores[0].score <= 1.0);
    }

    #[test]
    fn loop_matches_sklearn_reference_values() {
        let points = vec![vec![-1.1], vec![0.2], vec![101.1], vec![0.3]];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(123);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let scores = loop_outlier_scores(&tree, &data, 2, 2.0);

        let mut by_index = scores;
        by_index.sort_by(|a, b| {
            a.index
                .cmp(&b.index)
                .then_with(|| a.score.partial_cmp(&b.score).unwrap_or(Ordering::Equal))
        });

        assert!((by_index[0].score - 0.00314472).abs() < 1e-5);
        assert!(by_index[1].score.abs() < 1e-8);
        assert!((by_index[2].score - 0.68268573).abs() < 1e-5);
        assert!(by_index[3].score.abs() < 1e-8);
    }
}
