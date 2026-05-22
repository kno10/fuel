use std::cmp::Ordering;
use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Truncated k-means (t-kmeans) clustering aka k-means--.
///
/// A subset of the observations (controlled by `alpha`) is ignored ("trimmed")
/// when updating the cluster centers. This makes the algorithm robust to
/// outliers at the expense of ignoring a subset of the data.
///
/// `alpha` may be given either as a fraction (0 <= alpha < 1) or as an absolute
/// number of points to trim (alpha >= 1). When `alpha >= 1`, the algorithm
/// trims that many points; otherwise, it trims `alpha * n` points.
///
/// Internally the algorithm follows Lloyd's scheme but, on each iteration, it
/// keeps only the `keep` observations with the smallest squared distance to their
/// nearest center.
pub fn tkmeans<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N, alpha: f32,
) -> Result<KMeansResult<N>, String>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    assert!(k > 0);
    let alpha = if alpha < 0.0 { 0.0 } else { alpha }; // negative values treated as 0

    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);

    // Initialize cluster centers
    init.init::<A>(data, &mut cent, k);
    crate::check_interrupted()?;

    // pre-allocate scratch space
    let mut assign = vec![0_usize; n];
    let mut dist = vec![N::zero(); n];
    let mut kept = vec![true; n];
    let mut prev_assign = vec![usize::MAX; n];
    let mut prev_kept = vec![false; n];

    // Number of observations to keep (not trimmed).
    // If alpha >= 1 then alpha is treated as an absolute count of points to trim.
    // Otherwise it is a fraction of the dataset to trim.
    let keep = if alpha >= 1.0 {
        let trim = alpha.floor() as usize;
        n.saturating_sub(trim)
    } else {
        let frac = alpha.min(1.0);
        ((N::from(n).unwrap() * (N::one() - N::from(frac).unwrap())).floor())
            .to_usize()
            .unwrap_or(0)
            .min(n)
    };

    let mut iter = 0;

    // Auxiliary structures for center updates
    let mut sums = Centers::<N>::new(k, d);
    let mut counts = vec![0_usize; k];

    loop {
        iter += 1;

        // Assign points to nearest center and record squared distances
        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            let mut best = 0;
            let mut best_d = math::sqdist(cent.center(0), &scratch, d);
            for j in 1..k {
                let tmp = math::sqdist(cent.center(j), &scratch, d);
                if tmp < best_d {
                    (best, best_d) = (j, tmp);
                }
            }
            assign[i] = best;
            dist[i] = best_d;
        }

        // Determine which points are kept (not trimmed)
        select_keep(&dist, keep, &mut kept);

        // Convergence check: if both assignments and trim mask did not change,
        // we consider the algorithm converged.
        if iter > 1 && assign == prev_assign && kept == prev_kept {
            break;
        }

        // Update old state for next iteration
        prev_assign.clone_from(&assign);
        prev_kept.clone_from(&kept);

        // Save center state if we need to check tolerance
        let old_cent = cent.clone();
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };

        // Compute new centers using only kept points.
        // Reset accumulators from previous iterations.
        counts.fill(0);
        for j in 0..k {
            for v in sums.center_mut(j).iter_mut() {
                *v = N::zero();
            }
        }

        for i in 0..n {
            if !kept[i] {
                continue;
            }
            let a = assign[i];
            counts[a] += 1;
            data.load_into(i, &mut scratch, d);
            math::add_assign(sums.center_mut(a), &scratch, d);
        }

        for (j, &count) in counts.iter().enumerate() {
            if count > 0 {
                let inv = N::from(count).unwrap().recip();
                math::mul(cent.center_mut(j), sums.center(j), inv, d);
            } else {
                // if a cluster is empty (after trimming), set it to zero
                for v in cent.center_mut(j).iter_mut() {
                    *v = N::zero();
                }
            }
        }

        // Tolerance check
        if tol > N::zero() {
            let diff = cent.diff_frobenius_norm(&old_cent);
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                break;
            }
        }

        if iter >= maxiter {
            break;
        }
    }

    // compute objective on the full dataset so the reported inertia is
    // comparable to standard k-means (which uses all points, regardless of
    // trimming).
    let mut inertia = N::zero();
    for &d in &dist {
        inertia += d;
    }
    Ok(KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, inertia))
}

/// Select the smallest `keep` distances in `dist` and mark them as kept.
#[inline]
fn select_keep<N>(dist: &[N], keep: usize, kept: &mut [bool])
where
    N: Float,
{
    let n = dist.len();
    assert_eq!(n, kept.len());
    if keep >= n {
        kept.fill(true);
        return;
    }

    // Find the indices of the smallest `keep` elements.
    // Sorting is acceptable here since the dataset is typically not huge.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| dist[a].partial_cmp(&dist[b]).unwrap_or(Ordering::Equal));

    kept.fill(false);
    for &i in idx.iter().take(keep) {
        kept[i] = true;
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::util::{compute_loss, gen_test_data};

    #[test]
    fn test_tkmeans_equivalent_to_lloyd_when_alpha_zero() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = crate::cluster::kmeans::init::RandomSample::new(Pcg32::seed_from_u64(42));
        let mut init2 = crate::cluster::kmeans::init::RandomSample::new(Pcg32::seed_from_u64(42));

        let res_lloyd =
            crate::cluster::kmeans::lloyd::lloyd::<f64, _, _>(&dataset, 5, &mut init1, 100, 0.0)
                .unwrap();
        let res_tkmeans = tkmeans(&dataset, 5, &mut init2, 100, 0.0, 0.0).unwrap();

        assert_eq!(res_lloyd.iterations, res_tkmeans.iterations);
        assert_eq!(res_lloyd.assignments, res_tkmeans.assignments);
        assert!((res_lloyd.inertia.unwrap() - res_tkmeans.inertia.unwrap()).abs() < 1e-12);
    }

    #[test]
    fn test_tkmeans_trims_points() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = crate::cluster::kmeans::init::RandomSample::new(Pcg32::seed_from_u64(42));

        let res = tkmeans(&dataset, 5, &mut init, 100, 0.0, 0.1).unwrap();
        // With trimming, the reported inertia should be <= full inertia
        let full_loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert!(res.inertia.unwrap() <= full_loss);
    }
}
