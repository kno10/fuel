use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Stochastic k-medians update step.
///
/// This is a direct port of the algorithm used in the `kGmedian` R package
/// implementation.  It traverses the dataset once, updating each centre with an
/// adaptive learning rate that depends on the current cluster size and the
/// distance between the point and its assigned centre.
///
/// NOTE: the original R implementation uses a MacQueen-style initialization
/// (random sampling of initial centres).
#[inline(always)]
fn sto_kmed<N, A>(
    data: &A, data_tot: &A, k: usize, init_centers: &Centers<N>, nc: &mut [N], gamma: N, alpha: N,
) -> (Centers<N>, Vec<usize>, Vec<N>, Vec<usize>)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
{
    let n = data.nrows();
    let ntot = data_tot.nrows();
    let d = data.ncols();

    let mut centers_rm = init_centers.clone();
    let mut centers_av = init_centers.clone();

    // `nc` is carried across outer epochs so the stochastic learning-rate
    // schedule keeps decaying instead of restarting every pass.
    // final assignment for the total dataset
    let mut assign = vec![0_usize; ntot];

    // normalized distance (divided by sqrt(d)) to match the original R code.
    let p_sqrt = N::from(d).unwrap().sqrt();

    let mut scratch = vec![N::zero(); d];

    // Single pass over the data to update centres stochastically.
    for i in 0..n {
        data.load_into(i, &mut scratch, d);

        // assign to nearest centre (Euclidean distance)
        let mut best = math::sqdist(centers_rm.center(0), &scratch, d).sqrt() / p_sqrt;
        let mut best_j = 0;
        for j in 1..k {
            let tmp = math::sqdist(centers_rm.center(j), &scratch, d).sqrt() / p_sqrt;
            if tmp < best {
                best = tmp;
                best_j = j;
            }
        }

        // update the chosen cluster centre
        if best > N::zero() {
            // weight is gamma / (nc^alpha * sqrt(dist))
            let denom = nc[best_j].powf(alpha) * best.sqrt();
            if denom > N::zero() {
                let poids = gamma / denom;
                // delta = (x - centre) * poids
                let mut delta = scratch.clone();
                math::sub_assign(&mut delta, centers_rm.center(best_j), d);
                math::mul_assign(&mut delta, poids, d);
                math::add_assign(centers_rm.center_mut(best_j), &delta, d);
            }
        }

        // update cluster size and running average of centres
        nc[best_j] += N::one();
        let inv_nc = N::one() / nc[best_j];
        let mut delta = centers_rm.center(best_j).to_vec();
        math::sub_assign(&mut delta, centers_av.center(best_j), d);
        math::mul_assign(&mut delta, inv_nc, d);
        math::add_assign(centers_av.center_mut(best_j), &delta, d);
    }

    // Compute final assignments and within-cluster sums of distances.
    let mut wss = vec![N::zero(); k];
    let mut sizes = vec![0_usize; k];
    for (i, assign_i) in assign.iter_mut().enumerate() {
        data_tot.load_into(i, &mut scratch, d);
        let mut best = math::sqdist(centers_av.center(0), &scratch, d).sqrt() / p_sqrt;
        let mut best_j = 0;
        for j in 1..k {
            let tmp = math::sqdist(centers_av.center(j), &scratch, d).sqrt() / p_sqrt;
            if tmp < best {
                best = tmp;
                best_j = j;
            }
        }
        *assign_i = best_j;
        sizes[best_j] += 1;
        wss[best_j] += best;
    }

    (centers_av, assign, wss, sizes)
}

/// Internal implementation of kGmedian with math backend dispatch.
#[inline(always)]
pub fn kgmedians<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N, gamma: N, alpha: N,
) -> Result<KMeansResult<N>, String>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let maxiter = if maxiter == 0 { 1 } else { maxiter };

    let mut centers = Centers::<N>::new(k, data.ncols());
    init.init::<A>(data, &mut centers, k);

    let mut iter = 0_usize;
    let mut assign_best = Vec::new();
    let mut loss = N::infinity();
    let mut nc = vec![N::one(); k];

    while iter < maxiter {
        iter += 1;
        let (next_centers, assign, wss, _sizes) =
            sto_kmed::<N, A>(data, data, k, &centers, &mut nc, gamma, alpha);
        let next_loss = wss.iter().cloned().sum::<N>();
        let rel_loss = if loss.is_finite() {
            let denom = if loss > N::one() { loss } else { N::one() };
            (loss - next_loss).abs() / denom
        } else {
            N::infinity()
        };
        let mut rel = N::infinity();
        if tol > N::zero() {
            let old_norm = centers.frobenius_norm();
            let mut diff_sq = N::zero();
            for j in 0..k {
                diff_sq += math::sqdist(centers.center(j), next_centers.center(j), data.ncols());
            }
            let diff = diff_sq.sqrt();
            rel = if old_norm == N::zero() { diff } else { diff / old_norm };
        }

        centers = next_centers;
        assign_best = assign;
        loss = next_loss;

        if tol > N::zero() && (rel <= tol || rel_loss <= tol) {
            break;
        }
    }

    Ok(KMeansResult::with_inertia(centers.into_ndarray(), assign_best, iter, loss))
}

/// kGmedian -- stochastic k-medians clustering.
///
/// This is an implementation of the `kGmedian` algorithm as described in the
/// original R package (using a stochastic gradient-like update for each centre).
///
/// The function keeps the same interface style as other clustering routines in
/// this crate.
///
/// # Parameters
/// - `gamma`: Constant controlling the magnitude of the descent steps.
/// - `alpha`: Rate of decrease of the descent steps as cluster sizes grow.

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_kgmedians_runs() {
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(42)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res = kgmedians(&dataset, 5, &mut init, 100, 0.0, 1.0, 0.75).unwrap();
        let (cent, assign, _niter, loss) =
            (res.centers, res.assignments, res.iterations, res.inertia.unwrap_or_default());

        // verify loss is consistent with computed Euclidean loss
        let mut scratch = vec![0.0_f64; dataset.ncols()];
        let mut manual_loss = 0.0_f64;
        let p_sqrt = (dataset.ncols() as f64).sqrt();
        for (i, &idx) in assign.iter().enumerate().take(dataset.nrows()) {
            dataset.load_into(i, &mut scratch, dataset.ncols());
            let row = cent.row(idx);
            let sq = math::sqdist(&scratch, row.as_slice().unwrap(), dataset.ncols());
            manual_loss += sq.sqrt() / p_sqrt;
        }
        assert!((manual_loss - loss).abs() < 1e-12);
    }
}
