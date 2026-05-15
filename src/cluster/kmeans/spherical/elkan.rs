use super::common::*;
use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::{Centers, KMeansResult};
use crate::{Float, VectorData as Dataset, math, par_zip_chunks_map_mut};

#[inline(always)]
fn recompute_separation<N>(cent: &Centers<N>, k: usize, d: usize, csim: &mut [N], ccsim: &mut [N])
where
    N: Float,
{
    csim.fill(N::zero());
    for i in 0..k {
        ccsim[i * k + i] = N::one();
    }
    for i in 1..k {
        for j in 0..i {
            let s = clamp_one(math::dot(cent.center(i), cent.center(j), d));
            let sq = sqrt_half_sim(s);
            ccsim[i * k + j] = sq;
            ccsim[j * k + i] = sq;
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
fn sph_elkan_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>, ccsim: &mut [N],
) -> (Vec<usize>, Vec<usize>, Vec<N>)
where
    N: Float,
    A: Dataset<N> + Sync,
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
    let mut csim = vec![N::zero(); k];
    recompute_separation(cent, k, d, &mut csim, ccsim);
    let ccsim: &[N] = ccsim;
    let deltas: Vec<(Vec<usize>, Vec<N>)> =
        par_zip_chunks_map_mut(&mut assign, &mut bounds, k, |i0, assign_chunk, bounds_chunk| {
            let mut delta_csize = vec![0usize; k];
            let mut delta_sums = vec![N::zero(); k * d];
            let mut point = vec![N::zero(); d];
            for (ci, (aa, bounds_i)) in
                assign_chunk.iter_mut().zip(bounds_chunk.chunks_exact_mut(k)).enumerate()
            {
                let i = i0 + ci;
                data.load_into(i, &mut point, d);
                let mut best = clamp_one(math::dot(&point, cent.center(0), d));
                bounds_i[0] = best;
                let mut a = 0usize;
                for j in 1..k {
                    if best < ccsim[a * k + j] {
                        let sim = clamp_one(math::dot(&point, cent.center(j), d));
                        bounds_i[j] = sim;
                        if sim > best {
                            a = j;
                            best = sim;
                        }
                    } else {
                        bounds_i[j] = N::from(2).unwrap();
                    }
                }
                for j in 1..k {
                    if bounds_i[j] == N::from(2).unwrap() {
                        let cc = ccsim[a * k + j];
                        let simcc = cc * cc * N::from(2).unwrap() - N::one();
                        bounds_i[j] = sim_upper_bound(best, simcc);
                    }
                }
                *aa = a;
                delta_csize[a] += 1;
                math::add_assign(&mut delta_sums[a * d..a * d + d], &point, d);
            }
            (delta_csize, delta_sums)
        });
    for (dc, ds) in deltas {
        for j in 0..k {
            csize[j] += dc[j];
            math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
        }
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
pub fn spherical_elkan<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
where
    N: Float,
    I: Initialization<N>,
    A: Dataset<N> + Sync,
{
    let d = data.ncols();
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let mut msim = vec![N::one(); k];
    let mut csim = vec![N::zero(); k];
    let mut ccsim = vec![N::zero(); k * k];
    let (mut assign, mut csize, mut bounds) =
        sph_elkan_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums, &mut ccsim);
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
        recompute_separation(&cent, k, d, &mut csim, &mut ccsim);
        let ccsim: &[N] = &ccsim;
        let csim: &[N] = &csim;
        let deltas: Vec<(usize, Vec<N>, Vec<i64>)> = par_zip_chunks_map_mut(
            &mut assign,
            &mut bounds,
            k,
            |i0, assign_chunk, bounds_chunk| {
                let mut point = vec![N::zero(); d];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut delta_csize = vec![0i64; k];
                let mut local_changed = 0usize;
                for (ci, (aa, bounds_i)) in
                    assign_chunk.iter_mut().zip(bounds_chunk.chunks_exact_mut(k)).enumerate()
                {
                    let i = i0 + ci;
                    let orig = *aa;
                    let mut ls = bounds_i[orig];
                    if ls >= csim[orig] {
                        continue;
                    }
                    let mut recompute_ls = true;
                    let mut cur = orig;
                    for j in 0..k {
                        if j == orig || ls >= bounds_i[j] || ls >= ccsim[cur * k + j] {
                            continue;
                        }
                        if recompute_ls {
                            data.load_into(i, &mut point, d);
                            ls = clamp_one(math::dot(&point, cent.center(cur), d));
                            bounds_i[cur] = ls;
                            recompute_ls = false;
                            if ls >= bounds_i[j] || ls >= ccsim[cur * k + j] {
                                continue;
                            }
                        }
                        let sim = clamp_one(math::dot(&point, cent.center(j), d));
                        bounds_i[j] = sim;
                        if sim > ls {
                            cur = j;
                            ls = sim;
                        }
                    }
                    bounds_i[cur] = ls;
                    if cur != orig {
                        *aa = cur;
                        delta_csize[orig] -= 1;
                        delta_csize[cur] += 1;
                        math::sub_assign(&mut delta_sums[orig * d..orig * d + d], &point, d);
                        math::add_assign(&mut delta_sums[cur * d..cur * d + d], &point, d);
                        local_changed += 1;
                    }
                }
                (local_changed, delta_sums, delta_csize)
            },
        );
        let mut changed = 0;
        for (c, ds, dc) in deltas {
            changed += c;
            for j in 0..k {
                math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
                csize[j] = (csize[j] as i64 + dc[j]) as usize;
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
    use crate::cluster::kmeans::spherical::elkan::*;

    #[test]
    fn test_spherical_elkan_basic() {
        let mat = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.9, 0.1, -1.0, 0.0, -0.9, -0.1])
            .unwrap();
        let dataset = NdArrayDataset::new(&mat);
        let mut init = FirstK::new();
        let res = spherical_elkan(&dataset, 2, &mut init, 100, 0.0);
        assert!(res.iterations > 0, "spherical elkan did not run");
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
