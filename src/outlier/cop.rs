use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};
use rs_stats::Distribution;
use rs_stats::distributions::gamma_distribution::Gamma;

use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap, VectorData};

/// Probability distribution used for COP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopDistanceDist {
    ChiSquared,
    Gamma,
}

pub fn correlation_outlier_probabilities<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize, expect: f64, dist: CopDistanceDist,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "COP", false, F::zero(), F::zero(), F::one());
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::zero(); size],
            "COP",
            false,
            F::zero(),
            F::zero(),
            F::one(),
        );
    }

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);
    let dim = data.dims();

    let scores: Vec<F> = (0..size)
        .par_map(|idx| {
            let neigh = &neighborhoods[idx];
            if neigh.is_empty() {
                return F::zero();
            }

            let n = neigh.len();
            let mut centroid = vec![0.0_f64; dim];
            for (nb_idx, _) in neigh.iter() {
                let point = data.point(*nb_idx);
                for d in 0..dim {
                    centroid[d] += point[d].to_f64().unwrap_or(0.0);
                }
            }
            let nf = n as f64;
            centroid.iter_mut().for_each(|v| *v /= nf);

            if n < 2 || dim == 0 {
                return F::zero();
            }

            let mut points = Vec::with_capacity(n * dim);
            for (nb_idx, _) in neigh.iter() {
                let point = data.point(*nb_idx);
                for d in 0..dim {
                    let v = point[d].to_f64().unwrap_or(0.0);
                    points.push(v - centroid[d]);
                }
            }

            let mut cov_mat = Array2::<f64>::zeros((dim, dim));
            for i in 0..n {
                for r in 0..dim {
                    let vr = points[i * dim + r];
                    for c in 0..dim {
                        cov_mat[[r, c]] += vr * points[i * dim + c];
                    }
                }
            }
            let denom = nf; // population covariance to match ELKI StandardCovarianceMatrixBuilder
            cov_mat.iter_mut().for_each(|x| *x /= denom);

            let pca_res = match cov_mat.eigh(UPLO::Lower) {
                Ok((eigvals, eigvecs)) => (eigvals, eigvecs),
                Err(_) => return F::zero(),
            };

            let mut eigen_pairs: Vec<(f64, Vec<f64>)> = (0..dim)
                .map(|i| {
                    let val = pca_res.0[i];
                    let (vec, offset) = pca_res.1.column(i).to_owned().into_raw_vec_and_offset();
                    assert_eq!(offset, Some(0));
                    (val, vec)
                })
                .collect();
            eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut eigenvalues = Vec::with_capacity(dim);
            let mut eigenvectors = Vec::with_capacity(dim);
            for (evalue, evec) in eigen_pairs.into_iter() {
                eigenvalues.push(evalue.max(1e-12));
                eigenvectors.push(evec);
            }

            let point = data.point(idx);
            let mut diff = vec![0.0_f64; dim];
            for d in 0..dim {
                diff[d] = point[d].to_f64().unwrap_or(0.0) - centroid[d];
            }

            let mut projected = vec![0.0_f64; dim];
            for d in 0..dim {
                projected[d] = eigenvectors[d].iter().zip(diff.iter()).map(|(a, b)| a * b).sum();
            }

            if cfg!(debug_assertions) && idx < 10 {
                eprintln!("idx {} eigenvalues={:?} projected={:?}", idx, eigenvalues, projected);
            }

            let scores_by_dim: Vec<f64> = if matches!(dist, CopDistanceDist::ChiSquared) {
                let mut res = Vec::with_capacity(dim);
                let mut sqdevs = 0.0_f64;
                for d in 0..dim {
                    sqdevs += projected[d] * projected[d] / eigenvalues[d];
                    let chi = Gamma::new((d + 1) as f64 / 2.0, 0.5).unwrap();
                    let cdf = chi.cdf(sqdevs).unwrap_or(0.0);
                    res.push(1.0 - cdf);
                }
                res
            } else {
                // Gamma distribution estimation on scaled neighborhood distances
                let mut dists: Vec<Vec<f64>> = vec![vec![0.0_f64; n]; dim];
                for (j, (nb_idx, _)) in neigh.iter().enumerate() {
                    let point_nb = data.point(*nb_idx);
                    let mut sqdist = 0.0_f64;
                    for d in 0..dim {
                        let serrd = eigenvectors[d]
                            .iter()
                            .zip(
                                (0..dim).map(|k| point_nb[k].to_f64().unwrap_or(0.0) - centroid[k]),
                            )
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                        sqdist += serrd * serrd / eigenvalues[d];
                        dists[d][j] = sqdist;
                    }
                }
                let mut res = Vec::with_capacity(dim);
                let trim = ((0.85 * n as f64).floor() as usize).max(1).min(n);
                let mut sqdevs = 0.0_f64;
                for d in 0..dim {
                    sqdevs += projected[d] * projected[d] / eigenvalues[d];
                    let mut dcopy = dists[d].clone();
                    dcopy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let trimmed = &dcopy[..trim];

                    let gamma_dist = {
                        let cleaned: Vec<f64> =
                            trimmed.iter().copied().filter(|&v| v > 0.0 && v.is_finite()).collect();
                        if cleaned.is_empty() {
                            Gamma::new(1.0, 0.5).unwrap()
                        } else {
                            Gamma::fit(&cleaned).unwrap_or_else(|_| Gamma::new(1.0, 0.5).unwrap())
                        }
                    };

                    let cdf = gamma_dist.cdf(sqdevs).unwrap_or(0.0);
                    res.push(1.0 - cdf);
                }
                res
            };

            let min_score = scores_by_dim.iter().cloned().fold(f64::INFINITY, |acc, v| acc.min(v));
            let prob = expect * (1.0 - min_score) / (expect + min_score);
            let prob = prob.clamp(0.0, 1.0);
            if cfg!(debug_assertions) && idx < 10 {
                eprintln!(
                    "idx {} scores_by_dim={:?} min_score={} prob={}",
                    idx, scores_by_dim, min_score, prob
                );
            }
            F::from_f64(prob).unwrap_or(F::zero())
        });

    if cfg!(debug_assertions) {
        let sample: Vec<f64> = scores.iter().take(10).map(|x| x.to_f64().unwrap_or(0.0)).collect();
        eprintln!("COP first 10 scores: {:?}", sample);
    }

    make_outlier_result(scores, "COP", false, F::zero(), F::zero(), F::one())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    #[test]
    fn cop_remote_outlier_lowest() {
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

        let results =
            correlation_outlier_probabilities(&tree, &data, 2, 0.01, CopDistanceDist::ChiSquared);
        assert!(results.scores.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));

        let outlier_idx = points.len() - 1;
        let min_score = results.scores.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(results.scores[outlier_idx] <= min_score + 1e-12);
    }

    #[test]
    fn cop_10_matches_reference_outlier_score() {
        let points = crate::outlier::common::load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result =
            correlation_outlier_probabilities(&tree, &data, 10, 0.001, CopDistanceDist::Gamma);
        let reference = crate::outlier::common::load_reference_scores();
        let expected = reference.get("COP-10").expect("No reference for COP-10");
        let labels: Vec<u8> = crate::outlier::common::label_from_reference(&reference);

        if cfg!(debug_assertions) {
            for (i, (got, exp)) in result.scores.iter().zip(expected.iter()).enumerate().take(10) {
                println!("idx {} got {} expected {}", i, got, exp);
            }
        }

        crate::outlier::common::assert_outlier_auc_approx(
            "COP-10",
            crate::evaluation::outlier::receiver_operating_curve::auc(&result.scores, &labels),
            crate::evaluation::outlier::receiver_operating_curve::auc(expected, &labels),
            1e-6,
        );
        crate::outlier::common::assert_outlier_scores_approx(
            "COP-10",
            &result.scores,
            expected,
            1e-2, // less strict due to gamma distribution fitting variability
        );
    }
}
