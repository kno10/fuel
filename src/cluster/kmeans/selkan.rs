use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Perform the initial cluster assignment, recompute sums
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
fn simp_elkan_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
    scratch: &mut [N],
) -> (Vec<usize>, Vec<usize>, Vec<N>)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0_usize; n];
    let mut csize = vec![0_usize; k];
    let mut bounds = vec![N::zero(); n * k];
    // If possible, use the distances from initialization:
    if init.uses_distances() {
        init.init_with_distances::<A, _>(
            data,
            cent,
            k,
            Some(
                #[inline(always)]
                |j: usize, i: usize, d: N| {
                    bounds[i * k + j] = d.sqrt();
                },
            ),
        );
        for i in 0..n {
            let row = &bounds[i * k..i * k + k];
            let (mut b, mut bd) = (k, N::infinity());
            for (j, &distance) in row.iter().enumerate() {
                if distance < bd {
                    (b, bd) = (j, distance);
                }
            }
            assign[i] = b;
            csize[b] += 1;
            data.load_into(i, scratch, d);
            math::add_assign(sums.center_mut(b), scratch, d);
        }
    } else {
        init.init::<A>(data, cent, k);
        // Initial assignment, first iteration:
        for i in 0..n {
            data.load_into(i, scratch, d);
            let bounds_i = &mut bounds[i * k..i * k + k];
            let (mut a, mut s) = (0, N::infinity());
            for (j, bound_j) in bounds_i.iter_mut().enumerate() {
                let tmp = math::sqdist(cent.center(j), scratch, d).sqrt();
                *bound_j = tmp;
                if tmp < s {
                    (a, s) = (j, tmp);
                }
            }
            csize[a] += 1;
            assign[i] = a;
            math::add_assign(sums.center_mut(a), scratch, d);
        }
    }
    (assign, csize, bounds)
}

/// Simplified Elkan's algorithm
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn simp_elkan<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
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
    let (mut assign, mut csize, mut bounds) =
        simp_elkan_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums, &mut scratch);
    let mut iter = 1; // Initial iteration above!
    while iter < maxiter {
        iter += 1;
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };
        // Scale centers, compute movement
        let mut diff_sq = N::zero();
        for j in 0..k {
            if csize[j] > 0 {
                math::mul(&mut scratch, sums.center(j), N::from(csize[j]).unwrap().recip(), d);
                let movement = math::sqdist(&scratch, cent.center(j), d).sqrt();
                if tol > N::zero() {
                    diff_sq += movement * movement;
                }
                cmov[j] = movement;
                cent.center_mut(j).copy_from_slice(&scratch);
            } else {
                cmov[j] = N::zero();
            }
        }
        if tol > N::zero() {
            let diff = diff_sq.sqrt();
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                break;
            }
        }
        let mut changed = 0;
        for i in 0..n {
            let aa = assign[i];
            // Update bounds
            let bounds_i = &mut bounds[i * k..i * k + k];
            let mut upper_i = bounds_i[aa] + cmov[aa];
            math::sub_assign(bounds_i, &cmov, k); // we overwrite [aa] below!
            // Check bounds
            let (mut loaded, mut upper_tight, mut a) = (false, false, aa);
            for j in 0..k {
                if j == aa || upper_i <= bounds_i[j] {
                    continue;
                }
                // Make upper bound tight first:
                if !upper_tight {
                    if !loaded {
                        data.load_into(i, &mut scratch, d);
                        loaded = true;
                    }
                    upper_i = math::sqdist(cent.center(aa), &scratch, d).sqrt();
                    bounds_i[aa] = upper_i;
                    upper_tight = true;
                    if upper_i <= bounds_i[j] {
                        continue;
                    }
                }
                // Make lower tight
                bounds_i[j] = math::sqdist(cent.center(j), &scratch, d).sqrt();
                if bounds_i[j] < upper_i {
                    a = j;
                    upper_i = bounds_i[j];
                }
            }
            bounds_i[a] = upper_i; // store upper bound
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
    use crate::cluster::kmeans::ndarray::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let res = simp_elkan::<_, _, _>(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert_eq!(res.iterations, 11, "niter not as expected: {}", res.iterations);
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = simp_elkan::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);
        let n1 = res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = simp_elkan::<_, _, _>(&dataset, 5, &mut init2, 100, tol);
        let n2 = res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
