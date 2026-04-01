use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// K-harmonic means clustering as described in the literature.
///
/// The method maintains an objective function
/// `perf = \sum_i K / \sum_j d_{ij}^{-p}` and updates centers using weights
/// derived from both the membership and a per-sample weight.  The parameter
/// `p` controls the harmonic exponent (common choice `p=2`).  Convergence is
/// tested using the tolerance `tol` on the objective value.
///
/// # Arguments
/// * `data` – dataset
/// * `k` – number of clusters
/// * `init` – initialization strategy (center locations)
/// * `maxiter` – maximum number of iterations
/// * `tol` – convergence tolerance on the objective
/// * `p` – harmonic exponent (must be > 0)
///
/// The Rust API now orders `tol` before `p` to line up with the Python
/// bindings and make the parameters easier to supply when using kwargs.
pub fn kharmonic<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N, p: N,
) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    assert!(k > 0);
    assert!(p > N::zero());
    // note: `tol` was moved ahead of `p` in the signature above
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d]; // used for point loading
    let mut cent = Centers::<N>::new(k, d);

    // use provided initializer to set initial centers
    init.init::<A>(data, &mut cent, k);

    // storage for distances
    let mut dist = vec![N::zero(); n * k];

    let mut prev_perf = N::infinity();
    let mut iter = 0;
    let mut assignments = vec![0_usize; n];
    let mut loss = N::zero();

    while iter < maxiter {
        iter += 1;
        // compute distances (euclidean)
        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            for j in 0..k {
                let sd = math::sqdist(cent.center(j), &scratch, d);
                dist[i * k + j] = sd.sqrt();
            }
        }
        // performance
        let mut perf = N::zero();
        for i in 0..n {
            // harmonic mean component
            let mut hm = N::zero();
            for j in 0..k {
                let dij = dist[i * k + j];
                // avoid division by zero
                let inv = if dij > N::zero() { dij.powf(-p) } else { N::infinity() };
                hm += inv;
            }
            perf += N::from(k).unwrap() / hm;
        }
        // check convergence
        if (prev_perf - perf).abs() < tol {
            loss = perf;
            break;
        }
        prev_perf = perf;

        // compute membership and weights
        // membership M_ij = dist_ij^(-p-2) / sum_l dist_il^(-p-2)
        // weight W_i = sum_j dist_ij^(-p-2) / (sum_j dist_ij^-p)^2
        let mut membership = vec![N::zero(); n * k];
        let mut weight = vec![N::zero(); n];
        for i in 0..n {
            let mut num_sum = N::zero();
            let mut denom_sq = N::zero();
            // accumulate denominators
            for j in 0..k {
                let dij = dist[i * k + j];
                if dij > N::zero() {
                    let inv_p = dij.powf(-p);
                    let inv_p2 = dij.powf(-(p + N::from(2.0).unwrap()));
                    membership[i * k + j] = inv_p2; // temporarily store numerator
                    num_sum += inv_p2;
                    denom_sq += inv_p;
                } else {
                    membership[i * k + j] = N::zero();
                }
            }
            denom_sq = denom_sq * denom_sq;
            if denom_sq > N::zero() {
                weight[i] = num_sum / denom_sq;
            } else {
                weight[i] = N::zero();
            }
            // normalize membership
            if num_sum > N::zero() {
                for j in 0..k {
                    membership[i * k + j] = membership[i * k + j] / num_sum;
                }
            }
        }

        // update centers
        for j in 0..k {
            // reset sums
            for v in cent.center_mut(j).iter_mut() {
                *v = N::zero();
            }
            let mut denom = N::zero();
            for i in 0..n {
                let m_ij = membership[i * k + j];
                if m_ij == N::zero() {
                    continue;
                }
                data.load_into(i, &mut scratch, d);
                let w = m_ij * weight[i];
                // weighted accumulation: cent[j] += w * scratch
                math::axpy(cent.center_mut(j), w, &scratch, d);
                denom += w;
            }
            if denom > N::zero() {
                // divide the accumulated sum by denom to get the new center
                math::mul_assign(cent.center_mut(j), denom.recip(), d);
            }
        }
    }
    // final assignments and loss if not set
    if loss == N::zero() {
        loss = prev_perf;
    }
    for i in 0..n {
        let mut best = 0;
        let mut bestd = dist[i * k];
        for j in 1..k {
            let dij = dist[i * k + j];
            if dij < bestd {
                best = j;
                bestd = dij;
            }
        }
        assignments[i] = best;
    }
    KMeansResult::with_inertia(cent.into_ndarray(), assignments, iter, loss)
}

#[allow(dead_code)]
#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::cluster::kmeans::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(42)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res = kharmonic(&dataset, 5, &mut init, 100, 1e-6, 2.0);
        let (_cent, assign, niter, perf) =
            (res.centers, res.assignments, res.iterations, res.inertia.unwrap_or_default());
        assert_eq!(assign.len(), 100);
        assert!(perf.is_finite());
        assert!(niter <= 100);
    }
}
