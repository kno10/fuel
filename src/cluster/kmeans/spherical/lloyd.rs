use ndarray::Array2;

use crate::cluster::kmeans::Centers;
use crate::cluster::kmeans::init::*;
use crate::{Float, ParChunksMut, VectorData as Dataset, math};

/// Standard spherical k-means algorithm (Lloyd, Forgy with cosine similarity)
#[inline(always)]
pub fn spherical_lloyd<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> (Array2<N>, Vec<usize>, usize, N)
where
    N: Float,
    I: Initialization<N>,
    A: Dataset<N> + Sync,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    init.init::<A>(data, &mut cent, k);
    for j in 0..k {
        let nrm = math::dot(cent.center(j), cent.center(j), d).sqrt();
        if nrm > N::zero() {
            math::mul_assign(cent.center_mut(j), nrm.recip(), d);
        }
    }
    let mut assign = vec![0_usize; n];
    let mut csize = vec![0_usize; k];
    let mut lastsum = N::zero();
    let deltas: Vec<(Vec<usize>, Vec<N>, N)> =
        assign.as_mut_slice().par_chunks_map_mut(|i0, assign_chunk| {
            let mut delta_csize = vec![0usize; k];
            let mut delta_sums = vec![N::zero(); k * d];
            let mut local_sum = N::zero();
            let mut point = vec![N::zero(); d];
            for (ci, aa) in assign_chunk.iter_mut().enumerate() {
                let i = i0 + ci;
                data.load_into(i, &mut point, d);
                let (mut a, mut s) = (0, math::dot(&point, cent.center(0), d));
                for j in 1..k {
                    let tmp = math::dot(&point, cent.center(j), d);
                    if tmp > s {
                        (a, s) = (j, tmp);
                    }
                }
                *aa = a;
                delta_csize[a] += 1;
                math::add_assign(&mut delta_sums[a * d..a * d + d], &point, d);
                local_sum += s;
            }
            (delta_csize, delta_sums, local_sum)
        });
    for (dc, ds, ls) in deltas {
        for j in 0..k {
            csize[j] += dc[j];
            math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
        }
        lastsum += ls;
    }
    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
        // capture old centers if tolerance is enabled
        let old_cent = if tol > N::zero() { Some(cent.clone()) } else { None };
        // scale centers
        for (j, &csize_j) in csize.iter().enumerate().take(k) {
            if csize_j > 0 {
                math::mul(cent.center_mut(j), sums.center(j), N::from(csize_j).unwrap().recip(), d);
                let nrm = math::dot(cent.center(j), cent.center(j), d).sqrt();
                if nrm > N::zero() {
                    math::mul_assign(cent.center_mut(j), nrm.recip(), d);
                }
            }
        }
        // after updating centers check tolerance
        if let Some(ref old) = old_cent {
            let diff = cent.diff_frobenius_norm(old);
            let norm = old.frobenius_norm();
            let rel = if norm == N::zero() { diff } else { diff / norm };
            if rel <= tol {
                break;
            }
        }
        let deltas: Vec<(usize, N, Vec<N>, Vec<i64>)> =
            assign.as_mut_slice().par_chunks_map_mut(|i0, assign_chunk| {
                let mut point = vec![N::zero(); d];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut delta_csize = vec![0i64; k];
                let mut local_changed = 0usize;
                let mut local_sum = N::zero();
                for (ci, aa) in assign_chunk.iter_mut().enumerate() {
                    let i = i0 + ci;
                    let aa_old = *aa;
                    data.load_into(i, &mut point, d);
                    let (mut a, mut s) = (0, math::dot(&point, cent.center(0), d));
                    for j in 1..k {
                        let tmp = math::dot(&point, cent.center(j), d);
                        if tmp > s || (j == aa_old && tmp == s) {
                            (a, s) = (j, tmp);
                        }
                    }
                    local_sum += s;
                    if a != aa_old {
                        *aa = a;
                        delta_csize[aa_old] -= 1;
                        delta_csize[a] += 1;
                        math::sub_assign(&mut delta_sums[aa_old * d..aa_old * d + d], &point, d);
                        math::add_assign(&mut delta_sums[a * d..a * d + d], &point, d);
                        local_changed += 1;
                    }
                }
                (local_changed, local_sum, delta_sums, delta_csize)
            });
        let mut changed = 0;
        let mut sum = N::zero();
        for (c, s, ds, dc) in deltas {
            changed += c;
            sum += s;
            for j in 0..k {
                math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
                csize[j] = (csize[j] as i64 + dc[j]) as usize;
            }
        }
        lastsum = sum;
        if changed == 0 {
            break;
        }
    }
    (cent.into_ndarray(), assign, iter, -lastsum)
}

/// Spherical k-means clustering with the Standard Lloyd-style algorithm.
/// This maximizes cosine similarity and returns the negated total similarity.
#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use crate::NdArrayDataset;
    use crate::cluster::kmeans::init::FirstK;
    use crate::cluster::kmeans::spherical::lloyd::*;

    #[test]
    fn test_spherical_basic() {
        let mat = Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.9, 0.1, -1.0, 0.0, -0.9, -0.1])
            .unwrap();
        let dataset = NdArrayDataset::new(&mat);
        let mut init = FirstK::new();
        let (cent, assign, niter, loss) = spherical_lloyd(&dataset, 2, &mut init, 100, 0.0);
        assert!(niter > 0, "spherical lloyd did not run");
        assert!(loss <= 0.0, "expected negated similarity score");
        assert_eq!(assign[0], assign[1], "positive-direction points should match");
        assert_eq!(assign[2], assign[3], "negative-direction points should match");
        assert_ne!(assign[0], assign[2], "opposite directions should split");
        for j in 0..2 {
            let nrm = f64::sqrt(cent[[j, 0]] * cent[[j, 0]] + cent[[j, 1]] * cent[[j, 1]]);
            assert!((nrm - 1.0).abs() < 1e-12, "center is not normalized");
        }
    }
}
