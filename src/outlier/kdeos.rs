use rs_stats::Distribution;
use rs_stats::distributions::Normal;

use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
// Use for_each_knn(tree,data,k,true,...) when we need the self point included in neighbors.
use crate::outlier::kernel::KernelDensityFunction;
use crate::{DistanceData, Float, KnnSearch, ParMap, VectorData};

const KDEOS_CUTOFF: f64 = 1e-20;

pub fn kdeos<'a, S, D, F>(
    tree: &S, data: &'a D, kmin: usize, kmax: usize, kernel: KernelDensityFunction,
    min_bandwidth: f64, scale: f64, idim: Option<usize>,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "KDEOS",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let kmin = kmin.max(1);
    let kmax = kmax.max(kmin);
    let k_effective = kmax.min(size.saturating_sub(1));

    let knum = kmax + 1 - kmin;

    let neighborhoods: Vec<Vec<(usize, F)>> =
        for_each_knn(tree, data, k_effective + 1, true, |_, neigh| neigh);

    let dim = match idim {
        Some(d) if d > 0 => d as f64,
        _ => data.dims() as f64,
    };

    let iminbw = if min_bandwidth > 0.0 { 1.0 / (min_bandwidth * scale) } else { f64::INFINITY };

    let mut densities = vec![vec![0.0; knum]; size];

    for (_i, neigh) in neighborhoods.iter().enumerate().take(size) {
        if neigh.is_empty() {
            continue;
        }

        let mut sum_dist = 0.0;
        let mut idx_range = 0;

        for k in 1..=k_effective {
            let neighbor_dist = neigh[k - 1].1.to_f64().unwrap_or(f64::INFINITY);
            sum_dist += neighbor_dist;

            if k < kmin {
                continue;
            }

            let ibw = if sum_dist * scale > 0.0 {
                (k as f64) / (sum_dist * scale)
            } else {
                f64::INFINITY
            };
            let ibw = ibw.min(iminbw);

            let sca = ibw.powf(dim);
            for (neighbor_id, d) in neigh.iter() {
                let dval = d.to_f64().unwrap_or(0.0);
                let dens = if sca.is_finite() {
                    sca * kernel.density(dval * ibw)
                } else if dval == 0.0 {
                    1.0
                } else {
                    0.0
                };
                densities[*neighbor_id][idx_range] += dens;
                if dens < KDEOS_CUTOFF {
                    break;
                }
            }
            idx_range += 1;
            if idx_range >= knum {
                break;
            }
        }
    }

    let normal = Normal::new(0.0, 1.0).unwrap();
    let scores: Vec<F> = (0..size)
        .par_map(|i| {
            let neigh = &neighborhoods[i];
            let score: f64 = if neigh.is_empty() {
                1.0
            } else {
                let mut score_sum = 0.0_f64;
                // Include self and neighbors as in ELKI KDEOS score calculation:
                // query has been kept in neigh in position 0 from kNN(k_effective+1).
                for (k, _) in densities.iter().take(knum).enumerate() {
                    let mut mean = 0.0_f64;
                    for (neighbor_id, _) in neigh.iter() {
                        mean += densities[*neighbor_id][k];
                    }
                    mean /= neigh.len() as f64;

                    let mut variance = 0.0_f64;
                    for (neighbor_id, _) in neigh.iter() {
                        let dval = densities[*neighbor_id][k];
                        variance += (dval - mean).powi(2);
                    }

                    let stddev = if neigh.len() > 1 {
                        (variance / ((neigh.len() - 1) as f64)).sqrt()
                    } else {
                        0.0_f64
                    };

                    if stddev > 0.0_f64 {
                        score_sum += (mean - densities[i][k]) / stddev;
                    }
                }
                if knum > 0 {
                    score_sum /= knum as f64;
                }
                normal.cdf(score_sum).unwrap_or(0.0)
            };
            F::from_f64(score).unwrap_or(F::zero())
        });

    make_outlier_result(scores, "KDEOS", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;

    #[test]
    fn kdeos_remote_outlier() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results =
            kdeos(&tree, &data, 1, 2, KernelDensityFunction::Gaussian, 1e-6, 0.25, Some(1));
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }

    #[test]
    fn kdeos_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = kdeos(
            &tree,
            &data,
            10,
            10,
            KernelDensityFunction::Gaussian,
            0.0,
            0.5 * KernelDensityFunction::Gaussian.canonical_bandwidth(),
            Some(2),
        );
        let reference = load_reference_scores();
        let expected = reference.get("KDEOS-10").expect("No reference for KDEOS-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "KDEOS-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("KDEOS-10", &result.scores, expected, 1e-6);
    }
}
