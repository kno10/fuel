use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math, par_zip_chunks_map_mut};

/// Perform the initial cluster assignment, recompute sums
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
fn simp_elkan_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
) -> (Vec<usize>, Vec<usize>, Vec<N>)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N> + Sync,
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
        let deltas: Vec<(Vec<usize>, Vec<N>)> = par_zip_chunks_map_mut(
            &mut assign,
            &mut bounds,
            k,
            |i0, assign_chunk, bounds_chunk| {
                let mut delta_csize = vec![0usize; k];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut point = vec![N::zero(); d];
                for (ci, aa) in assign_chunk.iter_mut().enumerate() {
                    let i = i0 + ci;
                    let row = &bounds_chunk[ci * k..ci * k + k];
                    let (mut b, mut bd) = (k, N::infinity());
                    for (j, &distance) in row.iter().enumerate() {
                        if distance < bd {
                            (b, bd) = (j, distance);
                        }
                    }
                    *aa = b;
                    delta_csize[b] += 1;
                    data.load_into(i, &mut point, d);
                    math::add_assign(&mut delta_sums[b * d..b * d + d], &point, d);
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
        let centers = cent.as_ndarray();
        let mut matrix = vec![N::zero(); k.checked_mul(n).expect("point count overflow")];
        N::vec_pairwise_sqdist(centers, rows.view(), d, &mut matrix, k, n);
        let deltas: Vec<(Vec<usize>, Vec<N>)> = par_zip_chunks_map_mut(
            &mut assign,
            &mut bounds,
            k,
            |i0, assign_chunk, bounds_chunk| {
                let mut delta_csize = vec![0usize; k];
                let mut delta_sums = vec![N::zero(); k * d];
                for (ci, aa) in assign_chunk.iter_mut().enumerate() {
                    let i = i0 + ci;
                    let bounds_i = &mut bounds_chunk[ci * k..ci * k + k];
                    let (mut a, mut s) = (0, N::infinity());
                    for j in 0..k {
                        let tmp = matrix[j * n + i].sqrt();
                        bounds_i[j] = tmp;
                        if tmp < s {
                            (a, s) = (j, tmp);
                        }
                    }
                    delta_csize[a] += 1;
                    *aa = a;
                    math::add_assign(
                        &mut delta_sums[a * d..a * d + d],
                        rows.row(i).to_slice().unwrap(),
                        d,
                    );
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
    }
    (assign, csize, bounds)
}

/// Simplified Elkan's algorithm
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn simp_elkan<N, I, A>(
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
    let (mut assign, mut csize, mut bounds) =
        simp_elkan_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut sums);
    crate::check_interrupted()?;
    let mut iter = 1; // Initial iteration above!
    while iter < maxiter {
        iter += 1;
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };
        // Scale centers, compute movement
        let mut diff_sq = N::zero();
        for j in 0..k {
            if csize[j] > 0 {
                math::mul(&mut scratch, sums.center(j), N::from(csize[j]).unwrap().recip(), d);
                let movement = math::sqdist(&scratch, cent.center(j), d).sqrt();
                if tol > N::zero() {
                    diff_sq += movement * movement;
                }
                cmov[j] = movement;
                cent.center_mut(j).copy_from_slice(&scratch);
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
        let deltas: Vec<(usize, Vec<N>, Vec<i64>)> = par_zip_chunks_map_mut(
            &mut assign,
            &mut bounds,
            k,
            |i0, assign_chunk, bounds_chunk| {
                let mut scratch = vec![N::zero(); d];
                let mut delta_sums = vec![N::zero(); k * d];
                let mut delta_csize = vec![0i64; k];
                let mut local_changed = 0usize;
                for ci in 0..assign_chunk.len() {
                    let i = i0 + ci;
                    let aa = assign_chunk[ci];
                    let bounds_i = &mut bounds_chunk[ci * k..ci * k + k];
                    let mut upper_i = bounds_i[aa] + cmov[aa];
                    math::sub_assign(bounds_i, &cmov, k);
                    let (mut loaded, mut upper_tight, mut a) = (false, false, aa);
                    for j in 0..k {
                        if j == aa || upper_i <= bounds_i[j] {
                            continue;
                        }
                        if !upper_tight {
                            if !loaded {
                                data.load_into(i, &mut scratch, d);
                                loaded = true;
                            }
                            upper_i = math::sqdist(cent.center(aa), &scratch, d).sqrt();
                            bounds_i[aa] = upper_i;
                            upper_tight = true;
                            if upper_i <= bounds_i[j] {
                                continue;
                            }
                        }
                        bounds_i[j] = math::sqdist(cent.center(j), &scratch, d).sqrt();
                        if bounds_i[j] < upper_i {
                            a = j;
                            upper_i = bounds_i[j];
                        }
                    }
                    bounds_i[a] = upper_i;
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
        let res = simp_elkan::<_, _, _>(&dataset, 5, &mut init, 100, 0.0).unwrap();
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert_eq!(res.iterations, 11, "niter not as expected: {}", res.iterations);
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = simp_elkan::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0).unwrap();
        let n1 = res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = simp_elkan::<_, _, _>(&dataset, 5, &mut init2, 100, tol).unwrap();
        let n2 = res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
