use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

// helper alias to reduce complexity of return types in several algorithms
type AssignmentResult<N> = (Vec<usize>, Vec<usize>, Vec<(N, N)>, Vec<usize>);

/// Perform the initial cluster assignment, recompute sums
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub(crate) fn hamerly_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
    cdist: &mut [N], scratch: &mut [N],
) -> AssignmentResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0_usize; n];
    let mut assign2 = vec![0_usize; n]; // For Shallot algorithm, not used by Hamerly/Exponion
    let mut csize = vec![0_usize; k];
    let mut bounds = vec![(N::infinity(), N::infinity()); n];
    // If possible, use the distances from initialization:
    if init.uses_distances() {
        init.init_with_distances::<A, _>(
            data,
            cent,
            k,
            Some(
                #[inline(always)]
                |j: usize, i: usize, d: N| {
                    if d < bounds[i].0 {
                        (assign[i], assign2[i]) = (j, assign[i]);
                        bounds[i] = (d, bounds[i].0);
                    } else if d < bounds[i].1 {
                        assign2[i] = j;
                        bounds[i].1 = d;
                    }
                },
            ),
        );
        for i in 0..n {
            let a = assign[i];
            bounds[i] = (bounds[i].0.sqrt(), bounds[i].1.sqrt());
            csize[a] += 1;
            data.load_into(i, scratch, d);
            math::add_assign(sums.center_mut(a), scratch, d);
        }
    } else {
        init.init::<A>(data, cent, k);
        // Squared half center separation, d^2/4
        let mut idx = 0;
        for i in 1..k {
            let ci = &cent.center(i);
            for j in 0..i {
                debug_assert!(idx == triindex(i, j));
                cdist[idx] = N::from(0.25).unwrap() * math::sqdist(ci, cent.center(j), d);
                idx += 1;
            }
        }
        // Initial assignment, first iteration:
        for i in 0..n {
            data.load_into(i, scratch, d);
            let (mut a, mut s, mut b, mut s2) = (k, N::infinity(), k, N::infinity());
            for j in 0..k {
                if j <= 1 || s2 > cdist[triindex(a, j)] {
                    let tmp = math::sqdist(cent.center(j), scratch, d);
                    if tmp < s {
                        (a, s, b, s2) = (j, tmp, a, s);
                    } else if tmp < s2 {
                        (b, s2) = (j, tmp);
                    }
                }
            }
            csize[a] += 1;
            assign[i] = a;
            debug_assert!(b < k);
            assign2[i] = b;
            bounds[i] = (s.sqrt(), s2.sqrt());
            math::add_assign(sums.center_mut(a), scratch, d);
        }
    }
    (assign, csize, bounds, assign2)
}

/// Hamerly's algorithm
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn hamerly<N, I, A>(data: &A, k: usize, init: &mut I, maxiter: usize, tol: N) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let mut cmov = vec![N::zero(); k];
    let mut cdist = vec![N::zero(); (k * (k - 1)) >> 1]; // distances of centers * 0.5
    let mut cnear = vec![N::zero(); k];
    let (mut assign, mut csize, mut bounds, _) = hamerly_initial_assignment::<N, A, I>(
        data,
        k,
        init,
        &mut cent,
        &mut sums,
        &mut cdist,
        &mut scratch,
    );
    let mut iter = 1; // Initial iteration above!
    while iter < maxiter {
        iter += 1;
        // prepare old norm if required
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };
        // Scale centers, compute max movement
        let (mut most, mut cmov1, mut cmov2) = (0, N::zero(), N::zero());
        let mut diff_sq = N::zero();
        for j in 0..k {
            if csize[j] > 0 {
                math::mul(&mut scratch, sums.center(j), N::from(csize[j]).unwrap().recip(), d);
                let tmp = math::sqdist(&scratch, cent.center(j), d).sqrt();
                if tol > N::zero() {
                    diff_sq += tmp * tmp;
                }
                math::copy(cent.center_mut(j), &scratch, d);
                cmov[j] = tmp;
                if tmp > cmov1 {
                    (most, cmov1, cmov2) = (j, tmp, cmov1);
                } else if tmp > cmov2 {
                    cmov2 = tmp;
                }
            } else {
                cmov[j] = N::zero();
            }
        }
        // tolerance check
        if tol > N::zero() {
            let diff = diff_sq.sqrt();
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                break;
            }
        }
        // cluster separation, sqrt(d^2)/2
        cnear.fill(N::infinity());
        for i in 1..k {
            let ci = &cent.center(i);
            for j in 0..i {
                let tmp = math::sqdist(ci, cent.center(j), d);
                if tmp < cnear[i] {
                    cnear[i] = tmp;
                }
                if tmp < cnear[j] {
                    cnear[j] = tmp;
                }
            }
        }
        for value in cnear.iter_mut().take(k) {
            *value = N::from(0.5).unwrap() * value.sqrt();
        }
        let mut changed = 0;
        for i in 0..n {
            let aa = assign[i];
            // Update bounds
            let mut upper_i = bounds[i].0 + cmov[aa];
            let mut lower_i = bounds[i].1 - if aa != most { cmov1 } else { cmov2 };
            // Check bounds
            if upper_i <= lower_i || upper_i <= cnear[aa] {
                bounds[i] = (upper_i, lower_i); // update
                continue;
            }
            // Make upper bound tight first:
            data.load_into(i, &mut scratch, d);
            let daa = math::sqdist(cent.center(aa), &scratch, d); // squared
            upper_i = daa.sqrt(); // bounds are non-squared
            if upper_i <= lower_i || upper_i <= cnear[aa] {
                bounds[i] = (upper_i, lower_i); // update
                continue;
            }
            // Recompute other distances
            // Find two closest centers with distances
            let (mut a, mut s, mut b, mut s2) = (aa, daa, k, N::infinity());
            for j in 0..k {
                if j != aa {
                    let tmp = math::sqdist(cent.center(j), &scratch, d);
                    if tmp < s {
                        (a, s, b, s2) = (j, tmp, a, s);
                    } else if tmp < s2 {
                        (b, s2) = (j, tmp);
                    }
                }
            }
            // simpler: bounds[i] = (s.sqrt(), s2.sqrt());
            // We are lazy to call sqrt()
            // Compute lower first, as it needs the previous upper
            lower_i = if b == aa { upper_i } else { s2.sqrt() };
            upper_i = if a == aa { upper_i } else { s.sqrt() };
            bounds[i] = (upper_i, lower_i); // update
            if a != aa {
                assign[i] = a;
                csize[aa] -= 1;
                csize[a] += 1;
                math::sub_assign(sums.center_mut(aa), &scratch, d);
                math::add_assign(sums.center_mut(a), &scratch, d);
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }
    KMeansResult::without_inertia(cent.into_ndarray(), assign, iter)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let res = hamerly::<_, _, _>(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
        assert_eq!(res.iterations, 11, "niter not as expected");
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = hamerly::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);
        let n1 = res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = hamerly::<_, _, _>(&dataset, 5, &mut init2, 100, tol);
        let n2 = res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
