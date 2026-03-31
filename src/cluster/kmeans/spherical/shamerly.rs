use super::common::*;
use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::{Centers, KMeansResult};
use crate::math::DefaultMath;
use crate::math::Math;
use crate::{Float, VectorData as Dataset};
use std::iter::Sum;
use std::ops::*;

#[inline(always)]
fn sph_shamerly_initial_assignment<N, A, I>(
    data: &A,
    k: usize,
    init: &mut I,
    cent: &mut Centers<N>,
    sums: &mut Centers<N>,
) -> (Vec<usize>, Vec<usize>, Vec<(N, N)>)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0usize; n];
    let mut csize = vec![0usize; k];
    let mut bounds = vec![(N::zero(), -N::infinity()); n];
    init.init::<A>(data, cent, k);
    for j in 0..k {
        let nrm = DefaultMath::<N>::dot(cent.center(j), cent.center(j), d).sqrt();
        if nrm > N::zero() {
            DefaultMath::<N>::mul_assign(cent.center_mut(j), nrm.recip(), d);
        }
    }
    let mut scratch = vec![N::zero(); d];
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        let (mut a, mut s1, mut s2) = (
            0usize,
            clamp_one(DefaultMath::<N>::dot(&scratch, cent.center(0), d)),
            -N::infinity(),
        );
        for j in 1..k {
            let sim = clamp_one(DefaultMath::<N>::dot(&scratch, cent.center(j), d));
            if sim > s1 {
                (a, s1, s2) = (j, sim, s1);
            } else if sim > s2 {
                s2 = sim;
            }
        }
        assign[i] = a;
        bounds[i] = (s1, s2);
        csize[a] += 1;
        DefaultMath::<N>::add_assign(sums.center_mut(a), &scratch, d);
    }
    (assign, csize, bounds)
}

#[inline(always)]
fn update_bounds<N: Float>(bounds: &mut [(N, N)], assign: &[usize], msim: &[N]) {
    let mut least = 0usize;
    let mut delta = msim[0];
    let mut delta2 = N::one();
    for (i, &m) in msim.iter().enumerate().skip(1) {
        if m < delta {
            delta2 = delta;
            delta = m;
            least = i;
        } else if m < delta2 {
            delta2 = m;
        }
    }
    let dm = N::one() - delta * delta;
    let dm2 = N::one() - delta2 * delta2;
    for i in 0..bounds.len() {
        let ai = assign[i];
        let v2 = msim[ai];
        if v2 < N::one() {
            bounds[i].0 = sim_lower_bound(bounds[i].0, v2);
        }
        let w2 = if least == ai { dm2 } else { dm };
        if w2 > N::zero() {
            let w1 = clamp_one(bounds[i].1);
            let rad = (N::one() - w1 * w1) * w2;
            bounds[i].1 = w1
                + if rad > N::zero() {
                    rad.sqrt()
                } else {
                    N::zero()
                };
        }
    }
}

#[inline(always)]
pub fn spherical_shamerly<N, I, A>(
    data: &A,
    k: usize,
    init: &mut I,
    maxiter: usize,
    tol: N,
) -> KMeansResult<N>
where    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let mut msim = vec![N::one(); k];
    let (mut assign, mut csize, mut bounds) =
        sph_shamerly_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums);
    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
        // optionally remember old centers for tolerance check
        let old_cent = if tol > N::zero() {
            Some(cent.clone())
        } else {
            None
        };
        for j in 0..k {
            if csize[j] > 0 {
                DefaultMath::<N>::mul(
                    &mut scratch,
                    sums.center(j),
                    N::from(csize[j]).unwrap().recip(),
                    d,
                );
                let nrm = DefaultMath::<N>::norm(&scratch, d);
                if nrm > N::zero() {
                    DefaultMath::<N>::mul_assign(&mut scratch, nrm.recip(), d);
                    msim[j] = clamp_one(DefaultMath::<N>::dot(&scratch, cent.center(j), d));
                    DefaultMath::<N>::copy(cent.center_mut(j), &scratch, d);
                } else {
                    msim[j] = N::one();
                }
            } else {
                msim[j] = N::one();
            }
        }
        // check tolerance after updating centers
        if let Some(ref old) = old_cent {
            let diff = cent.diff_frobenius_norm(old);
            let norm = old.frobenius_norm();
            let rel = if norm == N::zero() { diff } else { diff / norm };
            if rel <= tol {
                break;
            }
        }
        update_bounds(&mut bounds, &assign, &msim);
        let mut changed = 0;
        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            let orig = assign[i];
            let mut ls = bounds[i].0;
            let us = bounds[i].1;
            if ls >= us {
                continue;
            }
            ls = clamp_one(DefaultMath::<N>::dot(&scratch, cent.center(orig), d));
            if ls >= us {
                bounds[i].0 = ls;
                continue;
            }
            let (mut cur, mut max2) = (orig, -N::infinity());
            for j in 0..k {
                if j == orig {
                    continue;
                }
                let sim = clamp_one(DefaultMath::<N>::dot(&scratch, cent.center(j), d));
                if sim > ls {
                    cur = j;
                    max2 = ls;
                    ls = sim;
                } else if sim > max2 {
                    max2 = sim;
                }
            }
            if cur != orig {
                assign[i] = cur;
                csize[orig] -= 1;
                csize[cur] += 1;
                DefaultMath::<N>::sub_assign(sums.center_mut(orig), &scratch, d);
                DefaultMath::<N>::add_assign(sums.center_mut(cur), &scratch, d);
                changed += 1;
            }
            bounds[i] = (ls, max2);
        }
        if changed == 0 {
            break;
        }
    }
    KMeansResult::without_inertia(cent.into_ndarray(), assign, iter)
}

pub fn spherical_simp_hamerly<N, I, A>(
    data: &A,
    k: usize,
    init: &mut I,
    maxiter: usize,
    tol: N,
) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display + 'static,
    I: Initialization<N>,
    A: Dataset<N>,
{
    spherical_shamerly::<N, I, A>(data, k, init, maxiter, tol)
}

#[cfg(test)]
mod tests {
    use crate::cluster::kmeans::init::FirstK;
    use crate::cluster::kmeans::ndarray::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;
    use super::*;
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    #[test]
    fn test_spherical_shamerly_basic() {
        let mat = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.9, 0.1, -1.0, 0.0, -0.9, -0.1])
            .unwrap();
        let dataset = NdArrayDataset::new(&mat);
        let mut init = FirstK::new();
        let res = crate::cluster::kmeans::spherical_simp_hamerly(&dataset, 2, &mut init, 100, 0.0);
        assert!(
            res.iterations > 0,
            "spherical simplified hamerly did not run"
        );
        assert_eq!(res.assignments.len(), 4);
        assert_eq!(
            res.assignments
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            2,
            "expected both clusters to be used"
        );
        for j in 0..2 {
            let nrm = ((res.centers[[j, 0]] as f64) * (res.centers[[j, 0]] as f64)
                + (res.centers[[j, 1]] as f64) * (res.centers[[j, 1]] as f64))
            .sqrt();
            assert!((nrm - 1.0).abs() < 1e-12, "center is not normalized");
        }
    }

    #[test]
    fn test_spherical_shamerly_tolerance() {
        // small dataset; tolerance should not increase iterations
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(42)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res1 = crate::cluster::kmeans::spherical_simp_hamerly(&dataset, 5, &mut init1, 100, 0.0);
        let (_c1, _a1, n1) = (res1.centers, res1.assignments, res1.iterations);
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res2 = crate::cluster::kmeans::spherical_simp_hamerly(&dataset, 5, &mut init2, 100, tol);
        let (_c2, _a2, n2) = (res2.centers, res2.assignments, res2.iterations);
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
