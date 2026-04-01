use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Perform the Lloyd cluster assignment
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub(crate) fn lloyd_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
    scratch: &mut [N],
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
    // If possible, use the distances from initialization:
    if init.uses_distances() {
        let mut best = vec![N::infinity(); n];
        init.init_with_distances::<A, _>(
            data,
            cent,
            k,
            Some(
                #[inline(always)]
                |j: usize, i: usize, d: N| {
                    if d < best[i] {
                        (assign[i], best[i]) = (j, d);
                    }
                },
            ),
        );
        for i in 0..n {
            let a = assign[i];
            csize[a] += 1;
            data.load_into(i, scratch, d);
            math::add_assign(sums.center_mut(a), scratch, d);
            lastsum += best[i];
        }
    } else {
        init.init::<A>(data, cent, k);
        // Initial assignment, first iteration:
        for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
            data.load_into(i, scratch, d);
            let (mut a, mut s) = (0, math::sqdist(cent.center(0), scratch, d));
            for j in 1..k {
                let tmp = math::sqdist(cent.center(j), scratch, d);
                if tmp < s {
                    (a, s) = (j, tmp);
                }
            }
            csize[a] += 1;
            *assign_i = a;
            math::add_assign(sums.center_mut(a), scratch, d);
            lastsum += s;
        }
    }
    (assign, csize, lastsum)
}

/// Standard k-means algorithm (Lloyd, Forgy)
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn lloyd<N, I, A>(data: &A, k: usize, init: &mut I, maxiter: usize, tol: N) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let (mut assign, mut csize, mut lastsum) =
        lloyd_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums, &mut scratch);
    let mut iter = 1; // Initial iteration above!
    while iter < maxiter {
        iter += 1;
        // compute norm of old centers if tolerance is requested
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };
        // Scale centers and optionally accumulate diff
        let mut diff_sq = N::zero();
        for (j, &csize_j) in csize.iter().enumerate().take(k) {
            if csize_j > 0 {
                // compute new center in scratch first
                math::mul(&mut scratch, sums.center(j), N::from(csize_j).unwrap().recip(), d);
                if tol > N::zero() {
                    let tmp_sq = math::sqdist(cent.center(j), &scratch, d);
                    diff_sq += tmp_sq;
                }
                math::copy(cent.center_mut(j), &scratch, d);
            }
        }
        // tolerance check
        if tol > N::zero() {
            // diff_sq already computed using math::sqdist
            let diff = diff_sq.sqrt();
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                break;
            }
        }
        let (mut changed, mut sum) = (0, N::zero());
        for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
            let aa = *assign_i;
            data.load_into(i, &mut scratch, d);
            let (mut a, mut s) = (0, math::sqdist(cent.center(0), &scratch, d));
            for j in 1..k {
                let tmp = math::sqdist(cent.center(j), &scratch, d);
                if tmp < s || (j == aa && tmp == s) {
                    (a, s) = (j, tmp);
                }
            }
            if a != aa {
                *assign_i = a;
                csize[aa] -= 1;
                csize[a] += 1;
                math::sub_assign(sums.center_mut(aa), &scratch, d);
                math::add_assign(sums.center_mut(a), &scratch, d);
                changed += 1;
            }
            sum += s;
        }
        lastsum = sum;
        if changed == 0 {
            break;
        }
    }
    KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::cluster::kmeans::ndarray::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let res = lloyd::<_, _, _>(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert_eq!(res.iterations, 11, "niter not as expected");
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        // small dataset; tuning tolerance should only decrease or equal iterations
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = lloyd::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);
        let (_cent1, _assign1, niter1, _) =
            (res1.centers, res1.assignments, res1.iterations, res1.inertia.unwrap_or_default());

        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = lloyd::<_, _, _>(&dataset, 5, &mut init2, 100, tol);
        let (_cent2, _assign2, niter2, _) =
            (res2.centers, res2.assignments, res2.iterations, res2.inertia.unwrap_or_default());

        assert!(niter2 <= niter1, "tolerance should not increase iteration count");
    }
}
