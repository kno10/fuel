use std::collections::VecDeque;

use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, IndexQuery, KnnSearch, RangeSearch};

/// Compute Dynamic Window Outlier Factor (DWOF) scores.
///
/// Based on progressive radius expansion and clustering, returns higher scores for
/// points that remain isolated for longer.
pub fn dynamic_window_outlier_factor<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize, delta: f64,
) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + RangeSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "DWOF", false, F::zero(), F::zero(), F::infinity());
    }

    let k_effective = k.min(size.saturating_sub(1));

    // initial radii based on k-nearest neighbors
    let neighborhoods =
        crate::outlier::common::for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);

    let mut radii = vec![F::zero(); size];
    let mut absolute_min_dist = F::infinity();
    let mut min_avg_dist = F::infinity();

    for i in 0..size {
        let neigh = &neighborhoods[i];
        let r = neigh.len();
        if r < 2 {
            radii[i] = F::zero();
            continue;
        }

        let mut sum = F::zero();
        let mut count = 0_usize;

        for u in 0..r {
            for v in (u + 1)..r {
                let uidx = neigh[u].0;
                let vidx = neigh[v].0;
                let dist = data.distance(uidx, vidx);

                if dist > F::zero() && dist < absolute_min_dist {
                    absolute_min_dist = dist;
                }

                sum += &dist;
                count += 1;
            }
        }

        let current_mean =
            if count > 0 { sum / F::from_usize(count).unwrap_or(F::zero()) } else { F::zero() };
        radii[i] = current_mean;
        if current_mean < min_avg_dist {
            min_avg_dist = current_mean;
        }
    }

    if min_avg_dist > F::zero() && min_avg_dist.is_finite() {
        for radius in radii.iter_mut().take(size) {
            *radius = if radius.is_finite() {
                absolute_min_dist * *radius / min_avg_dist
            } else {
                F::infinity()
            };
        }
    } else {
        for radius in radii.iter_mut().take(size) {
            *radius = F::infinity();
        }
    }

    let mut old_sizes = vec![1_usize; size];
    let mut new_sizes = vec![1_usize; size];
    let mut score = vec![F::zero(); size];

    let delta_f = F::from_f64(delta).unwrap_or(F::one());

    let mut count_unmerged = size;

    while count_unmerged > 0 {
        for radius in radii.iter_mut().take(size) {
            *radius = *radius * delta_f;
        }

        let mut labels: Vec<u32> = vec![u32::MAX; size];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for i in 0..size {
            if labels[i] != u32::MAX {
                continue;
            }

            let cluster_id = clusters.len();
            clusters.push(Vec::new());

            let mut queue = VecDeque::new();
            queue.push_back(i);
            labels[i] = cluster_id as u32;
            clusters[cluster_id].push(i);

            while let Some(cur) = queue.pop_front() {
                let mut query = data.query();
                query.set_index(cur);
                let neighbors = tree.search_range(&query, radii[cur]);

                for neighbor in neighbors {
                    let idx = neighbor.index;
                    if idx == cur {
                        continue;
                    }

                    if labels[idx] == u32::MAX {
                        labels[idx] = cluster_id as u32;
                        clusters[cluster_id].push(idx);
                        queue.push_back(idx);
                    } else if labels[idx] != cluster_id as u32 {
                        let other_cluster_id = labels[idx] as usize;
                        let other_points = clusters[other_cluster_id].clone();
                        for p in other_points {
                            labels[p] = cluster_id as u32;
                            clusters[cluster_id].push(p);
                            queue.push_back(p);
                        }
                        clusters[other_cluster_id].clear();
                    }
                }
            }
        }

        new_sizes.fill(0);
        for i in 0..size {
            new_sizes[i] = clusters[labels[i] as usize].len();
        }

        count_unmerged = new_sizes.iter().filter(|&&x| x == 1).count();

        for i in 0..size {
            let old = old_sizes[i] as f64;
            let new_val = new_sizes[i] as f64;

            let delta_score = if new_val > 0.0 { (old - 1.0) / new_val } else { 0.0 };

            score[i] += &F::from_f64(delta_score).unwrap_or(F::zero());
        }

        old_sizes.copy_from_slice(&new_sizes);

        // Safety guard to avoid infinite loops for pathological values.
        if radii.iter().all(|r| !r.is_finite()) {
            break;
        }
    }

    make_outlier_result(score, "DWOF", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn dwof_remote_outlier_lowest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = dynamic_window_outlier_factor(&tree, &data, 2, 1.1);
        assert!(results.scores.iter().all(|v| v.is_finite() && *v >= 0.0));

        let outlier_idx = points.len() - 1;
        let min_score = results.scores.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(results.scores[outlier_idx] <= min_score + 1e-12);
    }

    #[test]
    fn dwof_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = dynamic_window_outlier_factor(&tree, &data, 10, 1.1);
        let reference = load_reference_scores();
        let expected = reference.get("DWOF-10").expect("No reference for DWOF-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "DWOF-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("DWOF-10", &result.scores, expected, 1e-6);
    }
}
