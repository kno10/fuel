use crate::outlier::common::{OutlierResult, for_each_range, make_outlier_result};
use crate::{DistanceData, Float, ParMap, RangeSearch, VectorData};

/// LOCI: Fast Outlier Detection Using the Local Correlation Integral.
fn build_critical_distances<F: Float>(
    neighbors: &[(usize, F)], rmax: f64, alpha: f64,
) -> Vec<(f64, usize)> {
    let mut cdist: Vec<(f64, usize)> = Vec::new();

    for (j, &(_, dist_f)) in neighbors.iter().enumerate() {
        let dist = dist_f.to_f64().unwrap_or(0.0);
        if dist > rmax {
            break;
        }

        let count = j + 1;
        let next_tied =
            j + 1 < neighbors.len() && neighbors[j + 1].1.to_f64().unwrap_or(0.0) == dist;
        if next_tied {
            continue;
        }

        cdist.push((dist, count));
        if (alpha - 1.0).abs() > f64::EPSILON {
            let ri = dist / alpha;
            if ri <= rmax {
                cdist.push((ri, usize::MAX));
            }
        }
    }

    cdist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut last_k = 0usize;
    for entry in cdist.iter_mut() {
        if entry.1 == usize::MAX {
            entry.1 = last_k;
        } else {
            last_k = entry.1;
        }
    }

    cdist
}

fn find_last_le(cdist: &[(f64, usize)], value: f64) -> Option<usize> {
    if cdist.is_empty() || cdist[0].0 > value {
        return None;
    }
    let mut lo = 0;
    let mut hi = cdist.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cdist[mid].0 > value {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Some(lo - 1)
}

pub fn local_correlation_integral<'a, S, D, F>(
    tree: &S, data: &'a D, rmax: F, nmin: usize, alpha: F,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: RangeSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "LOCI", false, F::zero(), F::zero(), F::infinity());
    }

    let neighs = for_each_range(tree, data, rmax, true, |_, neigh| neigh);

    let rmax_f = rmax.to_f64().unwrap_or(f64::INFINITY);
    let alpha_f = alpha.to_f64().unwrap_or(1.0);

    let mut cdist_vec: Vec<Vec<(f64, usize)>> = Vec::with_capacity(size);
    let mut sorted_neighs: Vec<Vec<(usize, F)>> = Vec::with_capacity(size);
    for n in neighs.iter().take(size) {
        let mut n = n.clone();
        n.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_neighs.push(n.clone());
        cdist_vec.push(build_critical_distances(&n, rmax_f, alpha_f));
    }

    let max_score: Vec<f64> = (0..size).par_map(|i| {
        let current = &sorted_neighs[i];
        let cdist = &cdist_vec[i];

        let (maxdist, maxneig) = cdist.last().map(|(d, k)| (*d, *k)).unwrap_or((0.0, 0));

        let max_neighbors: Vec<(usize, f64)> = current
            .iter()
            .map(|(idx, dist)| (*idx, dist.to_f64().unwrap_or(0.0)))
            .take_while(|(_, dist)| *dist <= maxdist)
            .collect();

        if maxneig < nmin {
            f64::INFINITY
        } else {
            let mut best_mdef = 0.0;

            for &(r, count_r) in cdist.iter() {
                if count_r < nmin {
                    continue;
                }

                let alpha_r = alpha_f * r;
                let n_alpha = find_last_le(cdist, alpha_r).map(|idx| cdist[idx].1).unwrap_or(0);

                let mut neighbor_n_alphas = Vec::with_capacity(count_r);
                for &(neighbor_idx, neighbor_dist) in max_neighbors.iter() {
                    if neighbor_dist > r {
                        break;
                    }
                    let neighbor_cdist = &cdist_vec[neighbor_idx];
                    let neighbor_n_alpha = find_last_le(neighbor_cdist, alpha_r)
                        .map(|idx| neighbor_cdist[idx].1)
                        .unwrap_or(0) as f64;
                    neighbor_n_alphas.push(neighbor_n_alpha);
                }

                if neighbor_n_alphas.is_empty() {
                    continue;
                }

                let nhat =
                    neighbor_n_alphas.iter().sum::<f64>() / (neighbor_n_alphas.len() as f64);
                let var = neighbor_n_alphas
                    .iter()
                    .map(|v| {
                        let diff = *v - nhat;
                        diff * diff
                    })
                    .sum::<f64>();
                let sigma = (var / (neighbor_n_alphas.len() as f64)).sqrt();

                let mdef = nhat - (n_alpha as f64);
                let mdefnorm = mdef / sigma;
                let mdefnorm = if mdefnorm.is_nan() { 0.0 } else { mdefnorm };

                if mdefnorm > best_mdef {
                    best_mdef = mdefnorm;
                }
            }

            best_mdef
        }
    });

    let result_f: Vec<F> = max_score
        .iter()
        .map(|&v| {
            if v.is_infinite() {
                F::infinity()
            } else if v.is_nan() {
                F::zero()
            } else {
                F::from_f64(v).unwrap_or(F::zero())
            }
        })
        .collect();

    make_outlier_result(result_f, "LOCI", false, F::zero(), F::zero(), F::infinity())
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
    fn loci_remote_outlier_lowest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = local_correlation_integral(&tree, &data, 10.0, 2, 0.5);
        assert!(results.scores.iter().all(|v| !v.is_nan() && *v >= 0.0));

        let outlier_idx = points.len() - 1;
        let max_score = results
            .scores
            .iter()
            .cloned()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            results.scores[outlier_idx].is_infinite()
                || results.scores[outlier_idx] >= max_score - 1e-12
        );
    }

    #[test]
    fn loci_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);
        let result = local_correlation_integral(&tree, &data, 0.2, 20, 0.5);

        let reference = load_reference_scores();
        let expected = reference.get("LOCI-r0.2").expect("No reference for LOCI-r0.2");
        let labels: Vec<u8> = label_from_reference(&reference);
        assert_outlier_auc_approx(
            "LOCI-r0.2",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("LOCI-r0.2", &result.scores, expected, 1e-6);
    }
}
