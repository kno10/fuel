use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Compute initial assignment for k-medians using the provided initializer.
/// Returns (assignments, cluster_sizes, total_loss).
#[inline(always)]
pub fn kmedians_initial_assignment<N, I, A>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, scratch: &mut [N],
) -> (Vec<usize>, Vec<usize>, N)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0_usize; n];
    let mut csize = vec![0_usize; k];
    let mut lastsum = N::zero();

    // ignore `uses_distances`; we always perform an explicit assignment
    init.init::<A>(data, cent, k);
    // initial assignment
    for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
        data.load_into(i, scratch, d);
        let mut a = 0;
        let mut best_l1 = math::l1dist(cent.center(0), scratch, d);
        for j in 1..k {
            let tmp = math::l1dist(cent.center(j), scratch, d);
            if tmp < best_l1 {
                best_l1 = tmp;
                a = j;
            }
        }
        csize[a] += 1;
        *assign_i = a;
        // compute Euclidean distance for reported loss
        let sq = math::sqdist(cent.center(a), scratch, d);
        lastsum += sq.sqrt();
    }

    (assign, csize, lastsum)
}

/// Run the plain k-medians algorithm, updating centers by taking the per-axis
/// median of the points assigned to each cluster.  Distance for assignment is
/// Manhattan (L1).
pub fn kmedians<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> Result<KMeansResult<N>, String>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let (mut assign, mut csize, mut lastsum) =
        kmedians_initial_assignment::<N, I, A>(data, k, init, &mut cent, &mut scratch);
    crate::check_interrupted()?;

    let mut iter = 1; // initial iteration done above
    while iter < maxiter {
        iter += 1;
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };

        // compute new centers by median
        let mut new_cent = Centers::<N>::new(k, d);
        // buffers for each cluster/dimension
        let mut buffers: Vec<Vec<N>> = Vec::with_capacity(k * d);
        for _ in 0..(k * d) {
            buffers.push(Vec::new());
        }
        for (i, &cl) in assign.iter().enumerate().take(n) {
            if csize[cl] == 0 {
                continue;
            }
            data.load_into(i, &mut scratch, d);
            for p in 0..d {
                buffers[cl * d + p].push(scratch[p]);
            }
        }
        for j in 0..k {
            if csize[j] > 0 {
                for p in 0..d {
                    let buf = &mut buffers[j * d + p];
                    buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    // median index is floor(len/2)
                    let med = buf[buf.len() / 2];
                    new_cent.center_mut(j)[p] = med;
                }
            } else {
                // keep old center
                math::copy(new_cent.center_mut(j), cent.center(j), d);
            }
        }

        // tolerance check if requested
        if tol > N::zero() {
            let mut diff_sq = N::zero();
            for j in 0..k {
                diff_sq += math::sqdist(cent.center(j), new_cent.center(j), d);
            }
            let diff = diff_sq.sqrt();
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                cent = new_cent;
                break;
            }
        }

        cent = new_cent;

        let (mut changed, mut sum) = (0, N::zero());
        for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
            let aa = *assign_i;
            data.load_into(i, &mut scratch, d);
            // find cluster by L1 distance
            let mut a = 0;
            let mut best_l1 = math::l1dist(cent.center(0), &scratch, d);
            for j in 1..k {
                let tmp = math::l1dist(cent.center(j), &scratch, d);
                if tmp < best_l1 || (j == aa && tmp == best_l1) {
                    best_l1 = tmp;
                    a = j;
                }
            }
            if a != aa {
                *assign_i = a;
                csize[aa] -= 1;
                csize[a] += 1;
                changed += 1;
            }
            // record Euclidean distance for reporting
            let sq = math::sqdist(cent.center(a), &scratch, d);
            sum += sq.sqrt();
        }
        lastsum = sum;
        if changed == 0 {
            break;
        }
    }
    Ok(KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum))
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    fn euclidean_loss<N, A>(data: &A, centers: &Array2<N>, assign: &[usize]) -> N
    where
        N: Float + AddAssign + Sum + Copy,
        A: Dataset<N>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        let mut scratch = vec![N::zero(); d];
        let mut loss = N::zero();
        for (i, &idx) in assign.iter().enumerate().take(n) {
            data.load_into(i, &mut scratch, d);
            let cent = centers.row(idx);
            let mut sq = N::zero();
            for p in 0..d {
                let diff = scratch[p] - cent[p];
                sq += diff * diff;
            }
            loss += sq.sqrt();
        }
        loss
    }

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(42)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res = kmedians(&dataset, 5, &mut init, 100, 0.0).unwrap();
        let (cent, assign, niter, los) =
            (res.centers, res.assignments, res.iterations, res.inertia.unwrap_or_default());
        let loss: f64 = euclidean_loss(&dataset, &cent, &assign);
        assert!((loss - los).abs() < 1e-12, "loss not correct");
        // expected Euclidean loss from earlier run
        assert!((loss - 62.26872247515982).abs() < 1e-12, "loss not as expected: {}", loss);
        assert_eq!(niter, 7, "niter not as expected");
    }

    #[test]
    fn test_tolerance() {
        // small dataset; tolerance should not increase iterations
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(42)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res1 = kmedians(&dataset, 5, &mut init1, 100, 0.0).unwrap();
        let (_c1, _a1, n1, _) =
            (res1.centers, res1.assignments, res1.iterations, res1.inertia.unwrap_or_default());
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res2 = kmedians(&dataset, 5, &mut init2, 100, tol).unwrap();
        let (_c2, _a2, n2, _) =
            (res2.centers, res2.assignments, res2.iterations, res2.inertia.unwrap_or_default());
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
