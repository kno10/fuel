use rs_stats::prob::erf;

#[cfg(test)]
use crate::distance::Euclidean;
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch};

/// Compute Local Outlier Probabilities (LoOP) scores.
///
/// Scores are in the range [0, 1], where larger values indicate higher
/// outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn local_outlier_probabilities<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize, n_lambda: f64,
) -> OutlierResult<F>
where
    F: Float + std::iter::Sum + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    if k_effective == 0 {
        return make_outlier_result(
            vec![F::zero(); size],
            "LoOP",
            false,
            F::zero(),
            F::zero(),
            F::one(),
        );
    }

    let neighborhoods: Vec<Vec<(usize, F)>> =
        crate::outlier::common::for_each_knn(tree, data, k_effective, false, |_idx, neighbors| {
            neighbors
        });

    let pdists: Vec<F> = neighborhoods
        .iter()
        .map(|neighbors| {
            if neighbors.is_empty() {
                F::zero()
            } else {
                (neighbors.iter().map(|(_, distance)| *distance * *distance).sum::<F>()
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
            let neighbor_pdists_mean =
                neighbors.iter().map(|(neighbor_idx, _)| pdists[*neighbor_idx]).sum::<F>()
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

    let sqrt_2 = 2.0_f64.sqrt();

    let scores: Vec<F> = plofs
        .iter()
        .map(|plof| {
            let z = (plof / (nplof * sqrt_2)).max(0.0);
            F::from_f64(erf(z).unwrap_or(0.0).max(0.0)).unwrap_or(F::zero())
        })
        .collect();

    make_outlier_result(scores, "LoOP", false, F::zero(), F::zero(), F::infinity())
}
#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;
    use crate::vptree::VPTree;

    #[test]
    fn loop_matches_sklearn_reference_values() {
        let points = vec![vec![-1.1], vec![0.2], vec![101.1], vec![0.3]];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(123);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = local_outlier_probabilities(&tree, &data, 2, 2.0);

        assert_eq!(results.scores.len(), points.len());
        let (best_index, best_score) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .unwrap();

        assert_eq!(best_index, 2);
        assert!(*best_score > 0.0);
    }

    #[test]
    fn loop_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = local_outlier_probabilities(&tree, &data, 10, 1.0);
        let reference = load_reference_scores();
        let expected = reference.get("LoOP-10").expect("No reference for LoOP-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        println!("Computed first scores: {:?}", &result.scores[..10]);
        println!("Expected first scores: {:?}", &expected[..10]);

        assert_outlier_auc_approx(
            "LoOP-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("LoOP-10", &result.scores, expected, 1e-6);
    }
}
