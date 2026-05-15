use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, ParChunksMut, VectorData as Dataset, math};

/// Perform the Lloyd cluster assignment
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub(crate) fn lloyd_initial_assignment<N, A, I>(
    data: &A, rows: Option<ndarray::ArrayView2<'_, N>>, k: usize, init: &mut I,
    cent: &mut Centers<N>, sums: &mut Centers<N>,
) -> (Vec<usize>, Vec<usize>, N)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N> + Sync,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0_usize; n];
    let mut csize = vec![0_usize; k];
    let mut lastsum = N::zero();
    // If possible, use the distances from initialization:
    if init.uses_distances() {
        let mut best = vec![N::infinity(); n];
        init.init_with_distances::<A, _>(
            data,
            cent,
            k,
            Some(
                #[inline(always)]
                |j: usize, i: usize, d: N| {
                    if d < best[i] {
                        (assign[i], best[i]) = (j, d);
                    }
                },
            ),
        );
        let deltas: Vec<(Vec<usize>, Vec<N>, N)> =
            assign.as_mut_slice().par_chunks_map_mut(|i0, assign_chunk| {
                let mut delta_csize = vec![0usize; k];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut local_sum = N::zero();
                let mut point = vec![N::zero(); d];
                for (ci, &a) in assign_chunk.iter().enumerate() {
                    let i = i0 + ci;
                    delta_csize[a] += 1;
                    data.load_into(i, &mut point, d);
                    math::add_assign(&mut delta_sums[a * d..a * d + d], &point, d);
                    local_sum += best[i];
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
    } else {
        init.init::<A>(data, cent, k);
        let owned_rows;
        let rows = if let Some(rows) = rows {
            rows
        } else {
            owned_rows = data.to_ndarray();
            owned_rows.view()
        };
        let centers = cent.as_ndarray();
        let mut matrix = vec![N::zero(); k.checked_mul(n).expect("point count overflow")];
        // Use n*k layout (row = point, col = center) so the argmin scan over
        // centers is contiguous in memory.
        N::vec_pairwise_sqdist(rows, centers, d, &mut matrix, n, k);
        let deltas: Vec<(Vec<usize>, Vec<N>, N)> =
            assign.as_mut_slice().par_chunks_map_mut(|i0, assign_chunk| {
                let mut delta_csize = vec![0usize; k];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut local_sum = N::zero();
                for (ci, aa) in assign_chunk.iter_mut().enumerate() {
                    let i = i0 + ci;
                    let row = &matrix[i * k..i * k + k];
                    let mut a = 0;
                    let mut s = row[0];
                    for (j, &tmp) in row.iter().enumerate().skip(1) {
                        if tmp < s {
                            (a, s) = (j, tmp);
                        }
                    }
                    *aa = a;
                    delta_csize[a] += 1;
                    math::add_assign(
                        &mut delta_sums[a * d..a * d + d],
                        rows.row(i).to_slice().unwrap(),
                        d,
                    );
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
    }
    (assign, csize, lastsum)
}

/// Standard k-means algorithm (Lloyd, Forgy)
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn lloyd<N, I, A>(
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
    let (mut assign, mut csize, mut lastsum) =
        lloyd_initial_assignment::<N, A, I>(data, None, k, init, &mut cent, &mut sums);
    crate::check_interrupted()?;
    let mut iter = 1; // Initial iteration above!
    while iter < maxiter {
        iter += 1;
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };
        let mut diff_sq = N::zero();
        for (j, &csize_j) in csize.iter().enumerate().take(k) {
            if csize_j > 0 {
                math::mul(&mut scratch, sums.center(j), N::from(csize_j).unwrap().recip(), d);
                if tol > N::zero() {
                    diff_sq += math::sqdist(cent.center(j), &scratch, d);
                }
                math::copy(cent.center_mut(j), &scratch, d);
            }
        }
        if tol > N::zero() {
            let diff = diff_sq.sqrt();
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
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
                    let mut a = 0;
                    let mut s = math::sqdist(cent.center(0), &point, d);
                    for j in 1..k {
                        let tmp = math::sqdist(cent.center(j), &point, d);
                        if tmp < s {
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
        crate::check_interrupted()?;
        let mut changed = 0usize;
        let mut total_sum = N::zero();
        for (c, s, ds, dc) in deltas {
            changed += c;
            total_sum += s;
            for j in 0..k {
                math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
                csize[j] = (csize[j] as i64 + dc[j]) as usize;
            }
        }
        lastsum = total_sum;
        if changed == 0 {
            break;
        }
    }
    Ok(KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum))
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
        let res = lloyd::<_, _, _>(&dataset, 5, &mut init, 100, 0.0).unwrap();
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert_eq!(res.iterations, 11, "niter not as expected");
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        // small dataset; tuning tolerance should only decrease or equal iterations
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = lloyd::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0).unwrap();
        let (_cent1, _assign1, niter1, _) =
            (res1.centers, res1.assignments, res1.iterations, res1.inertia.unwrap_or_default());

        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = lloyd::<_, _, _>(&dataset, 5, &mut init2, 100, tol).unwrap();
        let (_cent2, _assign2, niter2, _) =
            (res2.centers, res2.assignments, res2.iterations, res2.inertia.unwrap_or_default());

        assert!(niter2 <= niter1, "tolerance should not increase iteration count");
    }

    /// End-to-end f32 test with d=8 (AVX2 path), k=10 (> MR_SDIST_F32=4).
    /// Reproduces the MNIST scenario dimensionality- and k-wise.
    /// lloyd must match lloyd_naive on same seed/data.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_d8_k10_matches_naive() {
        use ndarray::Array2;

        use crate::cluster::kmeans::lloyd_naive;

        // Generate f32 data: 200 points, d=8
        let mat_f64 = gen_test_data((200, 8), Pcg32::seed_from_u64(55));
        let mat_f32 = mat_f64.mapv(|x| x as f32);
        let dataset = NdArrayDataset::new(&mat_f32);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(55));
        let res1 = lloyd::<f32, _, _>(&dataset, 10, &mut init1, 200, 0.0).unwrap();

        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(55));
        let res2 = lloyd_naive::<f32, _, _>(&dataset, 10, &mut init2, 200, 0.0).unwrap();

        let loss1 = compute_loss(&dataset, &res1.centers, &res1.assignments);
        let loss2 = compute_loss(&dataset, &res2.centers, &res2.assignments);

        assert!(
            res1.iterations > 2,
            "lloyd converged in {} iterations - likely all-zero distance matrix",
            res1.iterations
        );
        assert_eq!(
            res1.iterations, res2.iterations,
            "lloyd={} naive={} iteration mismatch",
            res1.iterations, res2.iterations
        );
        assert!(
            (loss1 - loss2).abs() < 1e-3 * (loss2.abs() + 1.0),
            "loss mismatch: lloyd={loss1} naive={loss2}"
        );

        // Also verify inertia is non-zero (the MNIST failure symptom)
        assert!(loss1 > 0.0, "inertia is zero - distance matrix is all-zero");
        let _ = Array2::<f32>::zeros((1, 1)); // suppress unused import warning
    }

    /// Same as above but with d=784, matching MNIST dimensionality exactly.
    /// This catches bugs that only appear for large d in the AVX2 micro-kernel.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_f32_d784_k10_matches_naive() {
        use crate::cluster::kmeans::lloyd_naive;

        let mat_f64 = gen_test_data((200, 784), Pcg32::seed_from_u64(77));
        let mat_f32 = mat_f64.mapv(|x| x as f32);
        let dataset = NdArrayDataset::new(&mat_f32);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(77));
        let res1 = lloyd::<f32, _, _>(&dataset, 10, &mut init1, 200, 0.0).unwrap();

        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(77));
        let res2 = lloyd_naive::<f32, _, _>(&dataset, 10, &mut init2, 200, 0.0).unwrap();

        let loss1 = compute_loss(&dataset, &res1.centers, &res1.assignments);
        let loss2 = compute_loss(&dataset, &res2.centers, &res2.assignments);

        assert!(
            res1.iterations > 2,
            "lloyd converged in {} iterations - likely all-zero distance matrix",
            res1.iterations
        );
        assert_eq!(
            res1.iterations, res2.iterations,
            "lloyd={} naive={} iteration mismatch",
            res1.iterations, res2.iterations
        );
        assert!(
            (loss1 - loss2).abs() < 1e-3 * (loss2.abs() + 1.0),
            "loss mismatch: lloyd={loss1} naive={loss2}"
        );
        assert!(loss1 > 0.0, "inertia is zero");
    }
}
