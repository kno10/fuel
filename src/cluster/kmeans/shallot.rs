use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::hamerly::hamerly_initial_assignment;
use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Shallot algorithm
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn shallot<N, I, A>(data: &A, k: usize, init: &mut I, maxiter: usize, tol: N) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    // implementation unchanged
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let mut cmov = vec![N::zero(); k];
    let mut cdist = vec![N::zero(); (k * (k - 1)) >> 1]; // half (!) the distances of centers
    let mut cnear = vec![N::zero(); k]; // half (!) the distance to nearest other each
    let mut csort = vec![0_usize; k * (k - 1)]; // order of nearest centers
    for i in 0..k {
        for j in 0..i {
            csort[i * (k - 1) + j] = j;
        }
        for j in i + 1..k {
            csort[i * (k - 1) + j - 1] = j;
        }
    }
    let (mut assign, mut csize, mut bounds, mut assign2) = hamerly_initial_assignment::<N, A, I>(
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
        if tol > N::zero() {
            let diff = diff_sq.sqrt();
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                break;
            }
        }
        // cluster separation, sqrt(d^2)/2
        cnear.fill(N::infinity());
        let mut idx = 0;
        for i in 1..k {
            let ci = &cent.center(i);
            for j in 0..i {
                debug_assert!(idx == triindex(i, j));
                let tmp = N::from(0.5).unwrap() * math::sqdist(ci, cent.center(j), d).sqrt();
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
        // sort nearest centers. SortMeans and Exponion
        for i in 0..k {
            let slice = &mut csort[i * (k - 1)..(i + 1) * (k - 1)];
            // TODO: materialize into a scratch buffer? make cdist square?
            slice.sort_by(|&a, &b| {
                cdist[triindex(i, a)].partial_cmp(&cdist[triindex(i, b)]).unwrap()
            });
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
            let rhalf = upper_i + cnear[aa]; // cnear is already half
            // First other candidate from center
            if cdist[triindex(aa, csort[aa * (k - 1)])] > rhalf {
                continue;
            }
            // Shallot modification #1: try old second-nearest first:
            let bb = assign2[i];
            let dbb = math::sqdist(cent.center(bb), &scratch, d); // squared
            // Closest two known centers:
            let (mut ra, mut rb, mut dra2, mut drb2) = (aa, bb, daa, dbb);
            if dbb < daa {
                (ra, rb, dra2, drb2) = (bb, aa, dbb, daa);
                upper_i = dra2.sqrt();
            }
            // Shallot improvement 1.5:
            // note that db2 is still squared, cdist is half the distance
            // 0.5*(u+l), with l=min(u+d(x,p), 2u+2*cdist[z])
            let lp = upper_i + drb2.sqrt(); // l for p
            let lv = N::from(2).unwrap() * (upper_i + cdist[triindex(ra, csort[ra * (k - 1)])]); // l for v2(z)y
            let mut l = N::min(lp, lv);
            let mut rhalf = N::min(upper_i + cnear[ra], N::from(0.5).unwrap() * (upper_i + l));
            // Recompute other distances
            // Find closest center, and distance to two closest centers
            let (mut a, mut s) = (ra, dra2);
            let (mut b, mut s2) = (if lp < lv { rb } else { csort[ra * (k - 1)] }, l * l);
            for &j in &csort[ra * (k - 1)..(ra + 1) * (k - 1)] {
                if cdist[triindex(ra, j)] > rhalf {
                    break;
                }
                let tmp = if j == rb { drb2 } else { math::sqdist(cent.center(j), &scratch, d) };
                if tmp < s {
                    (a, s, b, s2) = (j, tmp, a, s);
                    if s2 < l * l {
                        // Second Shallot improvement: r shrinking
                        l = tmp.sqrt();
                        rhalf = N::min(rhalf, N::from(0.5).unwrap() * (upper_i + l));
                    }
                } else if tmp < s2 {
                    (b, s2) = (j, tmp);
                    // Second Shallot improvement: r shrinking
                    l = tmp.sqrt();
                    rhalf = N::min(rhalf, N::from(0.5).unwrap() * (upper_i + l));
                }
            }
            // simpler: bounds[i] = (s.sqrt(), s2.sqrt());
            // We are lazy to call sqrt()
            // Compute lower first, as it needs the previous upper
            lower_i = if b == aa { upper_i } else { s2.sqrt() };
            if a != aa {
                upper_i = s.sqrt()
            };
            bounds[i] = (upper_i, lower_i); // update
            if a != aa {
                assign[i] = a;
                csize[aa] -= 1;
                csize[a] += 1;
                math::sub_assign(sums.center_mut(aa), &scratch, d);
                math::add_assign(sums.center_mut(a), &scratch, d);
                changed += 1;
            }
            if b != bb {
                assert!(b < k);
                assign2[i] = b;
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
        let res = shallot::<_, _, _>(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert_eq!(res.iterations, 11, "niter not as expected");
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = shallot::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);
        let n1 = res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = shallot::<_, _, _>(&dataset, 5, &mut init2, 100, tol);
        let n2 = res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
