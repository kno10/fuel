use std::cmp::Ordering;

use crate::{DataAccess, DistanceFunction, MatrixDataAccess, VPTree};
#[cfg(test)]
use crate::EuclideanDistance;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopOutlierScore {
    pub index: usize,
    pub score: f64,
}

/// Compute Local Outlier Probabilities (LoOP) scores.
///
/// Scores are in the range [0, 1], where larger values indicate higher
/// outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn loop_outlier_scores<T>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, impl DistanceFunction<T>>,
    k: usize,
    n_lambda: f64,
) -> Vec<LoopOutlierScore> {
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    if k_effective == 0 {
        return vec![LoopOutlierScore {
            index: 0,
            score: 0.0,
        }];
    }

    let mut neighborhoods: Vec<Vec<(usize, f64)>> = vec![Vec::new(); size];

    for idx in data.iter() {
        let neighbors: Vec<(usize, f64)> = tree
            .search_knn(&data.with_query_index(idx), (k_effective + 1).min(size))
            .into_iter()
            .filter(|neighbor| neighbor.index() != idx)
            .take(k_effective)
            .map(|neighbor| (neighbor.index(), neighbor.distance()))
            .collect();

        neighborhoods[idx] = neighbors;
    }

    let pdists: Vec<f64> = neighborhoods
        .iter()
        .map(|neighbors| {
            if neighbors.is_empty() {
                0.0
            } else {
                (neighbors
                    .iter()
                    .map(|(_, distance)| distance * distance)
                    .sum::<f64>()
                    / neighbors.len() as f64)
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
                .sum::<f64>()
                / neighbors.len() as f64;

            if neighbor_pdists_mean > 0.0 {
                (pdists[idx] / neighbor_pdists_mean - 1.0).max(0.0)
            } else {
                0.0
            }
        })
        .collect();

    let mut nplof = n_lambda
        * (plofs.iter().map(|value| value * value).sum::<f64>()
            / plofs.len() as f64)
        .sqrt();

    if nplof <= 0.0 {
        nplof = 1.0;
    }

    let sqrt_2 = 2.0_f64.sqrt();

    let mut scores: Vec<LoopOutlierScore> = plofs
        .iter()
        .enumerate()
        .map(|(idx, plof)| LoopOutlierScore {
            index: idx,
            score: erf_approx((plof / (nplof * sqrt_2)).max(0.0)).max(0.0),
        })
        .collect();

    scores.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.index.cmp(&b.index))
    });

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
    let y = 1.0
        - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * (-(abs_x * abs_x)).exp());

    sign * y
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

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

        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(23);
        let tree = VPTree::new(&data, 2, &mut rng);

        let scores = loop_outlier_scores(&tree, &data, 2, 2.0);

        assert_eq!(scores.len(), points.len());
        assert_eq!(scores[0].index, 4);
        assert!(scores[0].score > 0.5);
        assert!(scores[0].score <= 1.0);
    }

    #[test]
    fn loop_matches_sklearn_reference_values() {
        let points = vec![vec![-1.1], vec![0.2], vec![101.1], vec![0.3]];

        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(123);
        let tree = VPTree::new(&data, 2, &mut rng);

        let scores = loop_outlier_scores(&tree, &data, 2, 2.0);

        let mut by_index = scores;
        by_index.sort_by(|a, b| {
            a.index.cmp(&b.index).then_with(|| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(Ordering::Equal)
            })
        });

        assert!((by_index[0].score - 0.00314472).abs() < 1e-5);
        assert!(by_index[1].score.abs() < 1e-8);
        assert!((by_index[2].score - 0.68268573).abs() < 1e-5);
        assert!(by_index[3].score.abs() < 1e-8);
    }
}
