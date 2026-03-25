use super::common::{OutlierResult, for_each_knn, make_outlier_result};
#[cfg(test)]
use crate::distance::EuclideanDistance;
use crate::{DistanceData, Float, KnnSearch};

/// Compute Local Outlier Factor (LOF) scores.
///
/// Scores around 1.0 indicate inlier behavior; scores significantly larger than
/// 1.0 indicate outlierness.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn local_outlier_factor<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
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
            vec![F::one(); size],
            "LOF",
            false,
            F::one(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods_and_k_distances: Vec<(Vec<(usize, F)>, F)> =
        for_each_knn(tree, data, k_effective, false, |_, neighbors| {
            let k_distance = neighbors.last().map_or(F::zero(), |(_, distance)| *distance);
            (neighbors, k_distance)
        });

    let neighborhoods: Vec<Vec<(usize, F)>> =
        neighborhoods_and_k_distances.iter().map(|(n, _)| n.clone()).collect();
    let k_distances: Vec<F> = neighborhoods_and_k_distances.iter().map(|(_, d)| *d).collect();

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

    let scores: Vec<F> = (0..size)
        .map(|idx| {
            let neighbors = &neighborhoods[idx];

            if neighbors.is_empty() || local_reachability_density[idx].is_infinite() {
                F::one()
            } else {
                let neighbor_lrd_sum = neighbors
                    .iter()
                    .map(|(neighbor_idx, _)| local_reachability_density[*neighbor_idx])
                    .sum::<F>();
                neighbor_lrd_sum
                    / (local_reachability_density[idx]
                        * F::from_usize(neighbors.len()).unwrap_or(F::zero()))
            }
        })
        .collect();

    make_outlier_result(scores, "LOF", false, F::zero(), F::zero(), F::infinity())
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
    fn lof_ranks_remote_point_highest() {
        let points =
            vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], vec![0.1, 0.1], vec![6.0, 6.0]];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(23);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = local_outlier_factor(&tree, &data, 2);

        assert_eq!(results.scores.len(), points.len());
        assert!(results.scores[4] > 1.0);

        assert!(
            results
                .scores
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != 4)
                .map(|(_, score)| *score)
                .all(|value| value > 0.5 && value < 2.0)
        );
    }

    #[test]
    fn lof_matches_sklearn_reference_values() {
        let points = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![2.0, 1.0]];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(123);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let _scores = local_outlier_factor(&tree, &data, 2);

        let results = local_outlier_factor(&tree, &data, 2);

        let sqrt2 = 2.0_f64.sqrt();
        let s0 = 2.0 * sqrt2 / (1.0 + sqrt2);
        let s1 = (1.0 + sqrt2) * (1.0 / (4.0 * sqrt2) + 1.0 / 2.0f64.mul_add(sqrt2, 2.0));

        assert!((results.scores[0] - s0).abs() < 1e-12);
        assert!((results.scores[1] - s1).abs() < 1e-12);
        assert!((results.scores[2] - s1).abs() < 1e-12);
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

        let scores = local_outlier_factor(&tree, &data, 5);
        assert_eq!(scores.scores.len(), points.len());

        let results = local_outlier_factor(&tree, &data, 5);

        let max_inlier = results.scores[..6].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_outlier = results.scores[6..].iter().copied().fold(f64::INFINITY, f64::min);

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
        let results = local_outlier_factor(&tree, &data, 5);

        assert!(results.scores.iter().all(|entry| !entry.is_nan()));

        let max_score = results.scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_score = results.scores.iter().copied().fold(f64::INFINITY, f64::min);

        assert!(max_score > 1.0);
        assert!(min_score < 1.0);
    }

    #[test]
    fn lof_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = local_outlier_factor(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("LOF-10").expect("No reference for LOF-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "LOF-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("LOF-10", &result.scores, expected, 1e-6);
    }
}
