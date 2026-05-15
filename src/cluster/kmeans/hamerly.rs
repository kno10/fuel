use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, ParChunksMut, VectorData as Dataset, math, par_zip_chunks_map_mut};

// helper alias to reduce complexity of return types in several algorithms
type AssignmentResult<N> = (Vec<usize>, Vec<usize>, Vec<(N, N)>, Vec<usize>);

/// Perform the initial cluster assignment, recompute sums
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub(crate) fn hamerly_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>, cdist: &mut [N],
) -> AssignmentResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N> + Sync,
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
        let deltas: Vec<(Vec<usize>, Vec<N>)> = par_zip_chunks_map_mut(
            &mut assign,
            &mut bounds,
            1,
            |i0, assign_chunk, bounds_chunk| {
                let mut delta_csize = vec![0usize; k];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut point = vec![N::zero(); d];
                for (ci, (&a, bound)) in
                    assign_chunk.iter().zip(bounds_chunk.iter_mut()).enumerate()
                {
                    let i = i0 + ci;
                    bound.0 = bound.0.sqrt();
                    bound.1 = bound.1.sqrt();
                    delta_csize[a] += 1;
                    data.load_into(i, &mut point, d);
                    math::add_assign(&mut delta_sums[a * d..a * d + d], &point, d);
                }
                (delta_csize, delta_sums)
            },
        );
        for (dc, ds) in deltas {
            for j in 0..k {
                csize[j] += dc[j];
                math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
            }
        }
    } else {
        init.init::<A>(data, cent, k);
        let rows = data.to_ndarray();
        // Squared half center separation, d^2/4 - needed for main loop
        let mut idx = 0;
        for i in 1..k {
            let ci = &cent.center(i);
            for j in 0..i {
                debug_assert!(idx == triindex(i, j));
                cdist[idx] = N::from(0.25).unwrap() * math::sqdist(ci, cent.center(j), d);
                idx += 1;
            }
        }
        let centers = cent.as_ndarray();
        let mut matrix = vec![N::zero(); k.checked_mul(n).expect("point count overflow")];
        N::vec_pairwise_sqdist(centers, rows.view(), d, &mut matrix, k, n);
        // Combined buffer: (assign, assign2, bounds) per point, written in parallel chunks.
        let mut combined = vec![(0usize, 0usize, (N::infinity(), N::infinity())); n];
        let deltas: Vec<(Vec<usize>, Vec<N>)> =
            combined.as_mut_slice().par_chunks_map_mut(|i0, chunk| {
                let mut delta_csize = vec![0usize; k];
                let mut delta_sums = vec![N::zero(); k * d];
                for (ci, item) in chunk.iter_mut().enumerate() {
                    let i = i0 + ci;
                    let (mut a, mut s, mut b, mut s2) = (k, N::infinity(), k, N::infinity());
                    for j in 0..k {
                        let tmp = matrix[j * n + i];
                        if tmp < s {
                            (a, s, b, s2) = (j, tmp, a, s);
                        } else if tmp < s2 {
                            (b, s2) = (j, tmp);
                        }
                    }
                    delta_csize[a] += 1;
                    debug_assert!(b < k);
                    *item = (a, b, (s.sqrt(), s2.sqrt()));
                    math::add_assign(
                        &mut delta_sums[a * d..a * d + d],
                        rows.row(i).to_slice().unwrap(),
                        d,
                    );
                }
                (delta_csize, delta_sums)
            });
        for (i, (a, b, bound)) in combined.into_iter().enumerate() {
            assign[i] = a;
            assign2[i] = b;
            bounds[i] = bound;
        }
        for (dc, ds) in deltas {
            for j in 0..k {
                csize[j] += dc[j];
                math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
            }
        }
    }
    (assign, csize, bounds, assign2)
}

/// Hamerly's algorithm
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn hamerly<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> Result<KMeansResult<N>, String>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N> + Sync,
{
    let d = data.ncols();
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let mut cmov = vec![N::zero(); k];
    let mut cdist = vec![N::zero(); (k * (k - 1)) >> 1]; // distances of centers * 0.5
    let mut cnear = vec![N::zero(); k];
    let (mut assign, mut csize, mut bounds, _) =
        hamerly_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums, &mut cdist);
    crate::check_interrupted()?;
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
        let deltas: Vec<(usize, Vec<N>, Vec<i64>)> = par_zip_chunks_map_mut(
            &mut assign,
            &mut bounds,
            1,
            |i0, assign_chunk, bounds_chunk| {
                let mut scratch = vec![N::zero(); d];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut delta_csize = vec![0i64; k];
                let mut local_changed = 0usize;
                for ci in 0..assign_chunk.len() {
                    let i = i0 + ci;
                    let aa = assign_chunk[ci];
                    let mut upper_i = bounds_chunk[ci].0 + cmov[aa];
                    let mut lower_i = bounds_chunk[ci].1 - if aa != most { cmov1 } else { cmov2 };
                    if upper_i <= lower_i || upper_i <= cnear[aa] {
                        bounds_chunk[ci] = (upper_i, lower_i);
                        continue;
                    }
                    data.load_into(i, &mut scratch, d);
                    let daa = math::sqdist(cent.center(aa), &scratch, d);
                    upper_i = daa.sqrt();
                    if upper_i <= lower_i || upper_i <= cnear[aa] {
                        bounds_chunk[ci] = (upper_i, lower_i);
                        continue;
                    }
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
                    lower_i = if b == aa { upper_i } else { s2.sqrt() };
                    upper_i = if a == aa { upper_i } else { s.sqrt() };
                    bounds_chunk[ci] = (upper_i, lower_i);
                    if a != aa {
                        assign_chunk[ci] = a;
                        delta_csize[aa] -= 1;
                        delta_csize[a] += 1;
                        math::sub_assign(&mut delta_sums[aa * d..aa * d + d], &scratch, d);
                        math::add_assign(&mut delta_sums[a * d..a * d + d], &scratch, d);
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
        crate::check_interrupted()?;
        if changed == 0 {
            break;
        }
    }
    Ok(KMeansResult::without_inertia(cent.into_ndarray(), assign, iter))
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
        let res = hamerly::<_, _, _>(&dataset, 5, &mut init, 100, 0.0).unwrap();
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
        let res1 = hamerly::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0).unwrap();
        let n1 = res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = hamerly::<_, _, _>(&dataset, 5, &mut init2, 100, tol).unwrap();
        let n2 = res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
