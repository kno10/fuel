use std::collections::HashSet;

use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

/// Influence Outlierness Factor.
pub fn influence_outlier<'a, S, D, F>(tree: &S, data: &'a D, k: usize, m: f64) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "INFLO", false, F::one(), F::zero(), F::infinity());
    }

    let k_effective = k.min(size.saturating_sub(1));

    let neighborhoods: Vec<Vec<(usize, F)>> =
        for_each_knn(tree, data, k_effective, false, |_idx, neighbors| neighbors);

    let mut rknn = vec![Vec::<usize>::new(); size];
    for (i, neigh) in neighborhoods.iter().enumerate() {
        for &(j, _) in neigh {
            if j < size {
                rknn[j].push(i);
            }
        }
    }

    let knn_distances: Vec<F> = neighborhoods
        .iter()
        .map(|neigh| neigh.last().map(|(_, d)| *d).unwrap_or(F::zero()))
        .collect();

    let mut pruned = vec![false; size];
    for (i, neigh) in neighborhoods.iter().enumerate() {
        let knn_set: HashSet<usize> = neigh.iter().map(|(idx, _)| *idx).collect();
        let r_set: HashSet<usize> = rknn[i].iter().copied().collect();
        let inter = knn_set.intersection(&r_set).count();

        let count = 1 + inter; // include the point itself
        if !knn_set.is_empty() && (count as f64) >= ((k_effective + 1) as f64) * m {
            pruned[i] = true;
        }
    }

    let scores: Vec<F> = (0..size)
        .par_map(|i| {
            if pruned[i] {
                F::one()
            } else {
                let valid_distance = |d: F| d.to_f64().filter(|v| v.is_finite() && *v > 0.0);
                let union: HashSet<usize> = neighborhoods[i]
                    .iter()
                    .map(|(j, _)| *j)
                    .chain(rknn[i].iter().copied())
                    .collect();

                let (sum, count) = union.iter().copied().filter(|&j| j != i).fold(
                    (0.0_f64, 0usize),
                    |(sum, count), j| match valid_distance(knn_distances[j]) {
                        Some(d_j) => (sum + 1.0 / d_j, count + 1),
                        None => (f64::INFINITY, count + 1),
                    },
                );

                let inflo = valid_distance(knn_distances[i])
                    .map(|kd| {
                        let val = sum * kd;
                        if val.is_nan() || val == 0.0 {
                            1.0
                        } else {
                            val / (count as f64)
                        }
                    })
                    .unwrap_or(1.0);

                F::from_f64(inflo).unwrap_or(F::zero())
            }
        });

    make_outlier_result(scores, "INFLO", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;

    #[test]
    fn inflo_test() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![10.0, 10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = influence_outlier(&tree, &data, 2, 1.0);
        assert_eq!(results.scores.len(), 4);
        assert!(results.scores[3] > results.scores[0]);
    }

    #[test]
    fn inflo_zero_distance_does_not_produce_nan() {
        let points = vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 1.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = influence_outlier(&tree, &data, 2, 1.0);
        assert_eq!(results.scores.len(), 3);
        assert!(results.scores.iter().all(|score| !score.is_nan()));
    }

    #[test]
    fn inflo_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = influence_outlier(&tree, &data, 10, 1.0);
        let reference = load_reference_scores();
        let expected = reference.get("INFLO-10").expect("No reference for INFLO-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "INFLO-10",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("INFLO-10", &result.scores, expected, 1e-6);
    }
}
