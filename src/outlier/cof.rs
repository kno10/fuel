use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

/// Compute Connectivity-based Outlier Factor (COF) scores.
///
/// Scores around 1.0 are expected for inliers; higher values indicate stronger outliers.
pub fn connectivity_outlier_factor<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "COF", false, F::one(), F::zero(), F::infinity());
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::one(); size],
            "COF",
            false,
            F::one(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);

    let acd: Vec<F> = (0..size).par_map(|i| {
        let neighbors = &neighborhoods[i];
        let r = neighbors.len();
        if r == 0 {
            return F::zero();
        }

        let k_plus = k_effective + 1;
        let mut candidate_indices: Vec<usize> = Vec::with_capacity(k_plus);
        let mut min_dists: Vec<Option<F>> = Vec::with_capacity(k_plus);

        candidate_indices.push(i);
        min_dists.push(None);

        for (neighbor_idx, _) in neighbors {
            candidate_indices.push(*neighbor_idx);
            min_dists.push(Some(data.distance(i, *neighbor_idx)));
        }

        let mut chain_sum = F::zero();

        for j in (1..k_plus).rev() {
            let mut min_pos = None;
            let mut min_dist = F::infinity();
            for (pos, &min_opt) in min_dists.iter().enumerate().take(k_plus) {
                if let Some(m) = min_opt
                    && m < min_dist
                {
                    min_dist = m;
                    min_pos = Some(pos);
                }
            }

            if let Some(pos) = min_pos {
                chain_sum += &(min_dist * F::from_usize(j).unwrap_or(F::zero()));
                let chosen = candidate_indices[pos];
                min_dists[pos] = None;

                for other in 0..k_plus {
                    if let Some(current) = min_dists[other] {
                        let dist = data.distance(chosen, candidate_indices[other]);
                        if dist < current {
                            min_dists[other] = Some(dist);
                        }
                    }
                }
            }
        }

        let denom = F::from_usize(k_plus).unwrap_or(F::zero())
            * F::from_f64(0.5).unwrap_or(F::zero())
            * F::from_usize(k_plus - 1).unwrap_or(F::zero());

        if denom > F::zero() { chain_sum / denom } else { F::zero() }
    });

    let k_plus_val = F::from_usize(k_effective + 1).unwrap_or(F::zero());
    let scores: Vec<F> = (0..size).par_map(|i| {
        let neighbors = &neighborhoods[i];
        if neighbors.is_empty() {
            return F::one();
        }
        let sum_neighbors: F = neighbors.iter().map(|(neighbor_idx, _)| acd[*neighbor_idx]).sum();
        if sum_neighbors > F::zero() {
            acd[i] * k_plus_val / sum_neighbors
        } else if acd[i] > F::zero() {
            F::infinity()
        } else {
            F::one()
        }
    });

    make_outlier_result(scores, "COF", false, F::one(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn cof_remote_outlier_highest() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = connectivity_outlier_factor(&tree, &data, 2);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        assert_eq!(best_index, 3);
    }

    #[test]
    fn cof_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = connectivity_outlier_factor(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("COF-10").expect("No reference for COF-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "COF-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("COF-10", &result.scores, expected, 1e-6);
    }
}
