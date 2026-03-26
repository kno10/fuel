use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, VectorData};

/// Subspace Outlier Degree approximation.
///
/// This implementation is adapted from ELKI's SOD logic with a simplified
/// neighborhood and subspace selection process.
pub fn subspace_outlier_degree<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize, alpha: f64,
) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.size();
    if size == 0 {
        return make_outlier_result(Vec::new(), "SOD", false, F::zero(), F::zero(), F::infinity());
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::zero(); size],
            "SOD",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    // Precompute Euclidean kNN neighborhoods (with self), in parallel.
    let euclidean_neighborhoods = for_each_knn(tree, data, k_effective, true, |_, neigh| {
        neigh.iter().map(|(idx, _)| *idx).collect::<Vec<usize>>()
    });

    let dim = data.dims();

    let scores: Vec<F> = (0..size)
        .map(|idx| {
            let query_neighbors = &euclidean_neighborhoods[idx];

            let mut similarities = Vec::with_capacity(size - 1);
            for (j, neighbor_list) in euclidean_neighborhoods.iter().enumerate().take(size) {
                if j == idx {
                    continue;
                }
                let shared =
                    query_neighbors.iter().filter(|&&v| neighbor_list.contains(&v)).count();
                similarities.push((j, shared));
            }

            similarities.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let neighborhood = if similarities.len() <= k_effective {
                similarities.into_iter().map(|(j, _)| j).collect::<Vec<usize>>()
            } else {
                let threshold = similarities[k_effective - 1].1;
                similarities
                    .into_iter()
                    .filter(|&(_, shared)| shared >= threshold)
                    .map(|(j, _)| j)
                    .collect::<Vec<usize>>()
            };

            if neighborhood.is_empty() {
                return F::zero();
            }

            let mut centroid = vec![0.0_f64; dim];
            for &nb_idx in &neighborhood {
                let point = data.point(nb_idx);
                for d in 0..dim {
                    centroid[d] += point[d].to_f64().unwrap_or(0.0);
                }
            }
            let n = neighborhood.len() as f64;
            centroid.iter_mut().for_each(|v| *v /= n);

            let mut variances = vec![0.0_f64; dim];
            for &nb_idx in &neighborhood {
                let point = data.point(nb_idx);
                for d in 0..dim {
                    let v = point[d].to_f64().unwrap_or(0.0);
                    let diff = v - centroid[d];
                    variances[d] += diff * diff;
                }
            }
            variances.iter_mut().for_each(|v| *v /= n);

            let mean_variance =
                if dim > 0 { variances.iter().sum::<f64>() / dim as f64 } else { 0.0 };
            let cutoff = alpha * mean_variance;

            let selected: Vec<usize> = variances
                .iter()
                .enumerate()
                .filter_map(|(d, v)| if *v < cutoff { Some(d) } else { None })
                .collect();

            if selected.is_empty() {
                return F::zero();
            }

            let center_point = data.point(idx);
            let sqr_dist = selected
                .iter()
                .map(|&d| {
                    let v = center_point[d].to_f64().unwrap_or(0.0);
                    let diff = v - centroid[d];
                    diff * diff
                })
                .sum::<f64>();

            let sod = if sqr_dist > 0.0 { sqr_dist.sqrt() / (selected.len() as f64) } else { 0.0 };

            F::from_f64(sod).unwrap_or(F::zero())
        })
        .collect();

    make_outlier_result(scores, "SOD", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::vptree::VPTree;

    #[test]
    fn sod_remote_outlier_highest() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = subspace_outlier_degree(&tree, &data, 2, 1.1);
        let (best_idx, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        assert_eq!(best_idx, 3);
    }

    #[test]
    fn sod_10_matches_reference_outlier_score() {
        let points = crate::outlier::common::load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = subspace_outlier_degree(&tree, &data, 10, 1.1);
        let reference = crate::outlier::common::load_reference_scores();
        let expected = reference.get("SOD-10").expect("No reference for SOD-10");
        let labels: Vec<u8> = crate::outlier::common::label_from_reference(&reference);

        let auc_result =
            crate::evaluation::outlier::receiver_operating_curve::auc(&result.scores, &labels);
        let auc_expected =
            crate::evaluation::outlier::receiver_operating_curve::auc(expected, &labels);
        println!("SOD-10: auc_result={}, auc_expected={}", auc_result, auc_expected);
        println!("SOD-10 first scores {:?}", &result.scores[0..10]);
        println!("SOD-10 first expected {:?}", &expected[0..10]);
        crate::outlier::common::assert_outlier_auc_approx("SOD-10", auc_result, auc_expected, 1e-6);
        crate::outlier::common::assert_outlier_scores_approx(
            "SOD-10",
            &result.scores,
            expected,
            1e-6,
        );
    }
}
