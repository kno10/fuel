use super::common::*;
use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::{Centers, KMeansResult};
use crate::math::DefaultMath;
use crate::math::Math;
use crate::{Float, VectorData as Dataset};
use std::iter::Sum;
use std::ops::*;

#[inline(always)]
fn recompute_separation<M, N>(cent: &Centers<N>, k: usize, d: usize, csim: &mut [N])
where    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
{
    csim.fill(N::zero());
    for i in 1..k {
        for j in 0..i {
            let s = clamp_one(DefaultMath::<N>::dot(cent.center(i), cent.center(j), d));
            let sq = sqrt_half_sim(s);
            if sq > csim[i] {
                csim[i] = sq;
            }
            if sq > csim[j] {
                csim[j] = sq;
            }
        }
    }
}

#[inline(always)]
fn sph_hamerly_initial_assignment<N, A, I>(
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
    let mut ccsim = vec![N::zero(); k * k];
    init.init::<A>(data, cent, k);
    for j in 0..k {
        let nrm = DefaultMath::<N>::dot(cent.center(j), cent.center(j), d).sqrt();
        if nrm > N::zero() {
            DefaultMath::<N>::mul_assign(cent.center_mut(j), nrm.recip(), d);
        }
    }
    for i in 0..k {
        ccsim[i * k + i] = N::one();
    }
    for i in 1..k {
        for j in 0..i {
            let s = clamp_one(DefaultMath::<N>::dot(cent.center(i), cent.center(j), d));
            let sq = sqrt_half_sim(s);
            ccsim[i * k + j] = sq;
            ccsim[j * k + i] = sq;
        }
    }
    let mut scratch = vec![N::zero(); d];
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        let mut max1 = clamp_one(DefaultMath::<N>::dot(&scratch, cent.center(0), d));
        let mut max2 = -N::infinity();
        let mut a = 0usize;
        for j in 1..k {
            if max2 < ccsim[a * k + j] {
                let sim = clamp_one(DefaultMath::<N>::dot(&scratch, cent.center(j), d));
                if sim > max1 {
                    a = j;
                    max2 = max1;
                    max1 = sim;
                } else if sim > max2 {
                    max2 = sim;
                }
            }
        }
        assign[i] = a;
        bounds[i] = (max1, max2);
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
pub fn spherical_hamerly<N, I, A>(
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
    let mut csim = vec![N::zero(); k];
    let (mut assign, mut csize, mut bounds) =
        sph_hamerly_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums);
    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
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
        update_bounds(&mut bounds, &assign, &msim);
        recompute_separation::<DefaultMath<N>, N>(&cent, k, d, &mut csim);
        let mut changed = 0;
        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            let orig = assign[i];
            let mut ls = bounds[i].0;
            let us = bounds[i].1;
            if ls >= us || ls >= csim[orig] {
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
        if let Some(ref old) = old_cent {
            let diff = cent.diff_frobenius_norm(old);
            let norm = old.frobenius_norm();
            let rel = if norm == N::zero() { diff } else { diff / norm };
            if rel <= tol {
                break;
            }
        }
    }
    KMeansResult::without_inertia(cent.into_ndarray(), assign, iter)
}




#[cfg(test)]
mod tests {
    use crate::cluster::kmeans::init::FirstK;
    use crate::cluster::kmeans::ndarray::NdArrayDataset;
    use crate::cluster::kmeans::spherical::hamerly::*;
    use ndarray::Array2;

    #[test]
    fn test_spherical_hamerly_basic() {
        let mat = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.9, 0.1, -1.0, 0.0, -0.9, -0.1])
            .unwrap();
        let dataset = NdArrayDataset::new(&mat);
        let mut init = FirstK::new();
        let res = spherical_hamerly(&dataset, 2, &mut init, 100, 0.0);
        assert!(res.iterations > 0, "spherical hamerly did not run");
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
        let cent = &res.centers;
        for j in 0..2 {
            let nrm = ((cent[[j, 0]] as f64) * (cent[[j, 0]] as f64)
                + (cent[[j, 1]] as f64) * (cent[[j, 1]] as f64)).sqrt();
            assert!((nrm - 1.0).abs() < 1e-12, "center is not normalized");
        }
    }
}
