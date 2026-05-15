use super::common::*;
use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::{Centers, KMeansResult};
use crate::{Float, VectorData as Dataset, math, par_zip_chunks_map_mut};

#[inline(always)]
fn recompute_separation<N>(cent: &Centers<N>, k: usize, d: usize, csim: &mut [N])
where
    N: Float,
{
    csim.fill(N::zero());
    for i in 1..k {
        for j in 0..i {
            let s = clamp_one(math::dot(cent.center(i), cent.center(j), d));
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
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
) -> (Vec<usize>, Vec<usize>, Vec<(N, N)>)
where
    N: Float,
    A: Dataset<N> + Sync,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0usize; n];
    let mut csize = vec![0usize; k];
    let mut bounds = vec![(N::zero(), -N::infinity()); n];
    let mut ccsim = vec![N::zero(); k * k];
    init.init::<A>(data, cent, k);
    for j in 0..k {
        let nrm = math::dot(cent.center(j), cent.center(j), d).sqrt();
        if nrm > N::zero() {
            math::mul_assign(cent.center_mut(j), nrm.recip(), d);
        }
    }
    for i in 0..k {
        ccsim[i * k + i] = N::one();
    }
    for i in 1..k {
        for j in 0..i {
            let s = clamp_one(math::dot(cent.center(i), cent.center(j), d));
            let sq = sqrt_half_sim(s);
            ccsim[i * k + j] = sq;
            ccsim[j * k + i] = sq;
        }
    }
    let ccsim: &[N] = &ccsim;
    let deltas: Vec<(Vec<usize>, Vec<N>)> =
        par_zip_chunks_map_mut(&mut assign, &mut bounds, 1, |i0, assign_chunk, bounds_chunk| {
            let mut delta_csize = vec![0usize; k];
            let mut delta_sums = vec![N::zero(); k * d];
            let mut point = vec![N::zero(); d];
            for (ci, (aa, bound)) in
                assign_chunk.iter_mut().zip(bounds_chunk.iter_mut()).enumerate()
            {
                let i = i0 + ci;
                data.load_into(i, &mut point, d);
                let mut max1 = clamp_one(math::dot(&point, cent.center(0), d));
                let mut max2 = -N::infinity();
                let mut a = 0usize;
                for j in 1..k {
                    if max2 < ccsim[a * k + j] {
                        let sim = clamp_one(math::dot(&point, cent.center(j), d));
                        if sim > max1 {
                            a = j;
                            max2 = max1;
                            max1 = sim;
                        } else if sim > max2 {
                            max2 = sim;
                        }
                    }
                }
                *aa = a;
                *bound = (max1, max2);
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
            bounds[i].1 = w1 + if rad > N::zero() { rad.sqrt() } else { N::zero() };
        }
    }
}

#[inline(always)]
pub fn spherical_hamerly<N, I, A>(
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
    let (mut assign, mut csize, mut bounds) =
        sph_hamerly_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums);
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
        update_bounds(&mut bounds, &assign, &msim);
        recompute_separation(&cent, k, d, &mut csim);
        let csim: &[N] = &csim;
        let deltas: Vec<(usize, Vec<N>, Vec<i64>)> = par_zip_chunks_map_mut(
            &mut assign,
            &mut bounds,
            1,
            |i0, assign_chunk, bounds_chunk| {
                let mut point = vec![N::zero(); d];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut delta_csize = vec![0i64; k];
                let mut local_changed = 0usize;
                for (ci, (aa, bound)) in
                    assign_chunk.iter_mut().zip(bounds_chunk.iter_mut()).enumerate()
                {
                    let i = i0 + ci;
                    let orig = *aa;
                    let mut ls = bound.0;
                    let us = bound.1;
                    if ls >= us || ls >= csim[orig] {
                        continue;
                    }
                    data.load_into(i, &mut point, d);
                    ls = clamp_one(math::dot(&point, cent.center(orig), d));
                    if ls >= us {
                        bound.0 = ls;
                        continue;
                    }
                    let (mut cur, mut max2) = (orig, -N::infinity());
                    for j in 0..k {
                        if j == orig {
                            continue;
                        }
                        let sim = clamp_one(math::dot(&point, cent.center(j), d));
                        if sim > ls {
                            cur = j;
                            max2 = ls;
                            ls = sim;
                        } else if sim > max2 {
                            max2 = sim;
                        }
                    }
                    if cur != orig {
                        *aa = cur;
                        delta_csize[orig] -= 1;
                        delta_csize[cur] += 1;
                        math::sub_assign(&mut delta_sums[orig * d..orig * d + d], &point, d);
                        math::add_assign(&mut delta_sums[cur * d..cur * d + d], &point, d);
                        local_changed += 1;
                    }
                    *bound = (ls, max2);
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
    use crate::cluster::kmeans::spherical::hamerly::*;

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
            res.assignments.iter().copied().collect::<std::collections::HashSet<_>>().len(),
            2,
            "expected both clusters to be used"
        );
        let cent = &res.centers;
        for j in 0..2 {
            let nrm = ((cent[[j, 0]] as f64) * (cent[[j, 0]] as f64)
                + (cent[[j, 1]] as f64) * (cent[[j, 1]] as f64))
                .sqrt();
            assert!((nrm - 1.0).abs() < 1e-12, "center is not normalized");
        }
    }
}
