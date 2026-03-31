use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::math::{DefaultMath, Math};
use crate::{Float, VectorData as Dataset};

/// Perform the initial cluster assignment, recompute sums
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
fn elkan_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
    scratch: &mut [N], cdist: &mut [N],
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
            DefaultMath::<N>::add_assign(sums.center_mut(b), scratch, d);
        }
    } else {
        init.init::<A>(data, cent, k);
        // half center separation, sqrt(d^2)/2
        let mut idx = 0;
        for i in 1..k {
            let ci = &cent.center(i);
            for j in 0..i {
                debug_assert!(idx == triindex(i, j));
                cdist[idx] =
                    N::from(0.5).unwrap() * DefaultMath::<N>::sqdist(ci, cent.center(j), d).sqrt();
                idx += 1;
            }
        }
        // Initial assignment, first iteration:
        for i in 0..n {
            data.load_into(i, scratch, d);
            let bounds_i = &mut bounds[i * k..i * k + k];
            let (mut a, mut s) = (0, N::infinity());
            for j in 0..k {
                if j == 0 || s > cdist[triindex(a, j)] {
                    let tmp = DefaultMath::<N>::sqdist(cent.center(j), scratch, d).sqrt();
                    bounds_i[j] = tmp;
                    if j == 0 || tmp < s {
                        (a, s) = (j, tmp);
                    }
                } else {
                    bounds_i[j] = N::nan(); // fill later
                }
            }
            // Fill skipped distances with bounds:
            for j in 1..k {
                if bounds_i[j].is_nan() {
                    bounds_i[j] = N::from(2).unwrap() * cdist[triindex(a, j)] - s;
                }
            }
            csize[a] += 1;
            assign[i] = a;
            DefaultMath::<N>::add_assign(sums.center_mut(a), scratch, d);
        }
    }
    (assign, csize, bounds)
}

/// Elkan's algorithm
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn elkan<N, I, A>(data: &A, k: usize, init: &mut I, maxiter: usize, tol: N) -> KMeansResult<N>
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
    let mut cdist = vec![N::zero(); (k * (k - 1)) >> 1];
    let mut cnear = vec![N::zero(); k];
    let (mut assign, mut csize, mut bounds) = elkan_initial_assignment::<N, A, I>(
        data,
        k,
        init,
        &mut cent,
        &mut sums,
        &mut scratch,
        &mut cdist,
    );
    let mut iter = 1; // Initial iteration above!
    while iter < maxiter {
        iter += 1;
        // prepare old norm for tolerance
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };
        // Scale centers, compute movement and accumulate diff
        let mut diff_sq = N::zero();
        for j in 0..k {
            if csize[j] > 0 {
                // copy the sum vector then scale it – this shows how the
                // `scale` helper can be used for simple scalar multiplication
                // without reimplementing an entire kernel.
                DefaultMath::<N>::copy(&mut scratch, sums.center(j), d);
                DefaultMath::<N>::scale(&mut scratch, N::from(csize[j]).unwrap().recip(), d);
                let movement = DefaultMath::<N>::sqdist(&scratch, cent.center(j), d).sqrt();
                if tol > N::zero() {
                    diff_sq += movement * movement;
                }
                cmov[j] = movement;
                DefaultMath::<N>::copy(cent.center_mut(j), &scratch, d);
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
        // use the math kernel rather than slice method so that
        // alternative implementations can override integer/packed
        // behaviour if needed.
        DefaultMath::<N>::fill(&mut cnear, N::infinity(), k);
        let mut idx = 0;
        for i in 1..k {
            let ci = &cent.center(i);
            for j in 0..i {
                debug_assert!(idx == triindex(i, j));
                let tmp =
                    N::from(0.5).unwrap() * DefaultMath::<N>::sqdist(ci, cent.center(j), d).sqrt();
                cdist[idx] = tmp;
                if tmp < cnear[i] {
                    cnear[i] = tmp;
                }
                if tmp < cnear[j] {
                    cnear[j] = tmp;
                }
                idx += 1;
            }
        }
        let mut changed = 0;
        for i in 0..n {
            let aa = assign[i];
            // Update bounds
            let bounds_i = &mut bounds[i * k..i * k + k];
            let mut upper_i = bounds_i[aa] + cmov[aa];
            DefaultMath::<N>::sub_assign(bounds_i, &cmov, k); // we overwrite [aa] below!
            if upper_i < cnear[aa] {
                bounds_i[aa] = upper_i; // store upper bound
                continue;
            }
            // Check bounds
            let (mut loaded, mut upper_tight, mut a) = (false, false, aa);
            for j in 0..k {
                if j == aa || upper_i <= bounds_i[j] || upper_i <= cdist[triindex(a, j)] {
                    continue;
                }
                // Make upper bound tight first:
                if !upper_tight {
                    if !loaded {
                        data.load_into(i, &mut scratch, d);
                        loaded = true;
                    }
                    upper_i = DefaultMath::<N>::sqdist(cent.center(aa), &scratch, d).sqrt();
                    bounds_i[aa] = upper_i;
                    upper_tight = true;
                    if upper_i <= bounds_i[j] || upper_i <= cdist[triindex(a, j)] {
                        continue;
                    }
                }
                // Make lower tight
                bounds_i[j] = DefaultMath::<N>::sqdist(cent.center(j), &scratch, d).sqrt();
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
                DefaultMath::<N>::sub_assign(sums.center_mut(aa), &scratch, d);
                DefaultMath::<N>::add_assign(sums.center_mut(a), &scratch, d);
                changed += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }
    // elkan does not compute inertia directly, return without it
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
        let result = elkan::<_, _, _>(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &result.centers, &result.assignments);
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
        assert_eq!(result.iterations, 11, "niter not as expected");
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let _res1 = elkan::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);
        let n1 = _res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let _res2 = elkan::<_, _, _>(&dataset, 5, &mut init2, 100, tol);
        let n2 = _res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
