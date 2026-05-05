use super::common::*;
use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::{Centers, KMeansResult};
use crate::{Float, VectorData as Dataset, math};

#[inline(always)]
fn sph_selkan_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
) -> (Vec<usize>, Vec<usize>, Vec<N>)
where
    N: Float,
    A: Dataset<N>,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0usize; n];
    let mut csize = vec![0usize; k];
    let mut bounds = vec![N::from(2).unwrap(); n * k];
    init.init::<A>(data, cent, k);
    for j in 0..k {
        let nrm = math::dot(cent.center(j), cent.center(j), d).sqrt();
        if nrm > N::zero() {
            math::mul_assign(cent.center_mut(j), nrm.recip(), d);
        }
    }
    let mut scratch = vec![N::zero(); d];
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        let bounds_i = &mut bounds[i * k..i * k + k];
        let mut a = 0;
        let mut best = clamp_one(math::dot(&scratch, cent.center(0), d));
        bounds_i[0] = best;
        for (j, bound_j) in bounds_i.iter_mut().enumerate().take(k).skip(1) {
            let sim = clamp_one(math::dot(&scratch, cent.center(j), d));
            *bound_j = sim;
            if sim > best {
                a = j;
                best = sim;
            }
        }
        assign[i] = a;
        csize[a] += 1;
        math::add_assign(sums.center_mut(a), &scratch, d);
    }
    (assign, csize, bounds)
}

#[inline(always)]
fn update_bounds<N: Float>(bounds: &mut [N], assign: &[usize], msim: &[N], k: usize) {
    for i in 0..assign.len() {
        let a = assign[i];
        let bi = &mut bounds[i * k..i * k + k];
        bi[a] = sim_lower_bound(bi[a], msim[a]);
        for j in 0..k {
            bi[j] = sim_upper_bound(bi[j], msim[j]);
        }
    }
}

#[inline(always)]
pub fn spherical_simp_elkan<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
where
    N: Float,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let mut msim = vec![N::one(); k];
    let (mut assign, mut csize, mut bounds) =
        sph_selkan_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums);
    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
        let old_cent = if tol > N::zero() { Some(cent.clone()) } else { None };
        for j in 0..k {
            if csize[j] > 0 {
                math::mul(&mut scratch, sums.center(j), N::from(csize[j]).unwrap().recip(), d);
                let nrm = math::norm(&scratch, d);
                if nrm > N::zero() {
                    math::mul_assign(&mut scratch, nrm.recip(), d);
                    msim[j] = clamp_one(math::dot(&scratch, cent.center(j), d));
                    math::copy(cent.center_mut(j), &scratch, d);
                } else {
                    msim[j] = N::one();
                }
            } else {
                msim[j] = N::one();
            }
        }
        update_bounds(&mut bounds, &assign, &msim, k);
        let mut changed = 0;
        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            let orig = assign[i];
            let bounds_i = &mut bounds[i * k..i * k + k];
            let mut ls = bounds_i[orig];
            let mut recompute_ls = true;
            let mut cur = orig;
            for j in 0..k {
                if j == orig || ls >= bounds_i[j] {
                    continue;
                }
                if recompute_ls {
                    ls = clamp_one(math::dot(&scratch, cent.center(cur), d));
                    bounds_i[cur] = ls;
                    recompute_ls = false;
                    if ls >= bounds_i[j] {
                        continue;
                    }
                }
                let sim = clamp_one(math::dot(&scratch, cent.center(j), d));
                bounds_i[j] = sim;
                if sim > ls {
                    cur = j;
                    ls = sim;
                }
            }
            bounds_i[cur] = ls;
            if cur != orig {
                assign[i] = cur;
                csize[orig] -= 1;
                csize[cur] += 1;
                math::sub_assign(sums.center_mut(orig), &scratch, d);
                math::add_assign(sums.center_mut(cur), &scratch, d);
                changed += 1;
            }
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
    use ndarray::Array2;

    use crate::NdArrayDataset;
    use crate::cluster::kmeans::init::FirstK;
    use crate::cluster::kmeans::spherical::selkan::*;

    #[test]
    fn test_spherical_selkan_basic() {
        let mat = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.9, 0.1, -1.0, 0.0, -0.9, -0.1])
            .unwrap();
        let dataset = NdArrayDataset::new(&mat);
        let mut init = FirstK::new();
        let res = spherical_simp_elkan(&dataset, 2, &mut init, 100, 0.0);
        assert!(res.iterations > 0, "spherical simplified elkan did not run");
        assert_eq!(res.assignments.len(), 4);
        assert_eq!(
            res.assignments.iter().copied().collect::<std::collections::HashSet<_>>().len(),
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
}
