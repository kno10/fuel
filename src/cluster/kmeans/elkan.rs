use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Perform the initial cluster assignment, recompute sums
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
fn elkan_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, sums: &mut Centers<N>,
    scratch: &mut [N], cdist: &mut [N],
) -> (Vec<usize>, Vec<usize>, Vec<N>)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
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
        for i in 0..n {
            let row = &bounds[i * k..i * k + k];
            let (mut b, mut bd) = (k, N::infinity());
            for (j, &distance) in row.iter().enumerate() {
                if distance < bd {
                    (b, bd) = (j, distance);
                }
            }
            assign[i] = b;
            csize[b] += 1;
            data.load_into(i, scratch, d);
            math::add_assign(sums.center_mut(b), scratch, d);
        }
    } else {
        init.init::<A>(data, cent, k);
        let rows = data.to_ndarray();
        // half center separation, sqrt(d^2)/2 - needed for main loop
        let mut idx = 0;
        for i in 1..k {
            let ci = &cent.center(i);
            for j in 0..i {
                debug_assert!(idx == triindex(i, j));
                cdist[idx] = N::from(0.5).unwrap() * math::sqdist(ci, cent.center(j), d).sqrt();
                idx += 1;
            }
        }
        let centers = cent.as_ndarray();
        let mut matrix = vec![N::zero(); k.checked_mul(n).expect("point count overflow")];
        N::vec_pairwise_sqdist(centers, rows.view(), d, &mut matrix, k, n);
        for i in 0..n {
            let bounds_i = &mut bounds[i * k..i * k + k];
            let (mut a, mut s) = (0, N::infinity());
            for j in 0..k {
                let tmp = matrix[j * n + i].sqrt();
                bounds_i[j] = tmp;
                if tmp < s {
                    (a, s) = (j, tmp);
                }
            }
            csize[a] += 1;
            assign[i] = a;
            math::add_assign(sums.center_mut(a), rows.row(i).to_slice().unwrap(), d);
        }
    }
    (assign, csize, bounds)
}

/// Elkan's algorithm
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
pub fn elkan<N, I, A>(data: &A, k: usize, init: &mut I, maxiter: usize, tol: N) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N> + Sync,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let mut cmov = vec![N::zero(); k];
    let mut cdist = vec![N::zero(); (k * (k - 1)) >> 1];
    let mut cnear = vec![N::zero(); k];
    let (mut assign, mut csize, mut bounds) = elkan_initial_assignment::<N, A, I>(
        data,
        k,
        init,
        &mut cent,
        &mut sums,
        &mut scratch,
        &mut cdist,
    );
    let mut iter = 1; // Initial iteration above!
    while iter < maxiter {
        iter += 1;
        // prepare old norm for tolerance
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };
        // Scale centers, compute movement and accumulate diff
        let mut diff_sq = N::zero();
        for j in 0..k {
            if csize[j] > 0 {
                // copy the sum vector then scale it – this shows how the
                // `scale` helper can be used for simple scalar multiplication
                // without reimplementing an entire kernel.
                math::copy(&mut scratch, sums.center(j), d);
                math::scale(&mut scratch, N::from(csize[j]).unwrap().recip(), d);
                let movement = math::sqdist(&scratch, cent.center(j), d).sqrt();
                if tol > N::zero() {
                    diff_sq += movement * movement;
                }
                cmov[j] = movement;
                math::copy(cent.center_mut(j), &scratch, d);
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
        // use the math kernel rather than slice method so that
        // alternative implementations can override integer/packed
        // behaviour if needed.
        math::fill(&mut cnear, N::infinity(), k);
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
        let changed = 'iter: {
            #[cfg(feature = "parallel")]
            if n >= crate::math::PARALLEL_ROW_THRESHOLD {
                use rayon::prelude::*;
                let chunk_size = n.div_ceil(rayon::current_num_threads());
                let deltas: Vec<(usize, Vec<N>, Vec<i64>)> = assign
                    .par_chunks_mut(chunk_size)
                    .zip(bounds.par_chunks_mut(chunk_size * k))
                    .enumerate()
                    .map(|(ti, (assign_chunk, bounds_chunk))| {
                        let i0 = ti * chunk_size;
                        let chunk_n = assign_chunk.len();
                        let mut scratch = vec![N::zero(); d];
                        let mut delta_sums = vec![N::zero(); k * d];
                        let mut delta_csize = vec![0i64; k];
                        let mut local_changed = 0usize;
                        for ci in 0..chunk_n {
                            let i = i0 + ci;
                            let aa = assign_chunk[ci];
                            let bounds_i = &mut bounds_chunk[ci * k..ci * k + k];
                            let mut upper_i = bounds_i[aa] + cmov[aa];
                            math::sub_assign(bounds_i, &cmov, k);
                            if upper_i < cnear[aa] {
                                bounds_i[aa] = upper_i;
                                continue;
                            }
                            let (mut loaded, mut upper_tight, mut a) = (false, false, aa);
                            for j in 0..k {
                                if j == aa
                                    || upper_i <= bounds_i[j]
                                    || upper_i <= cdist[triindex(a, j)]
                                {
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
                                    if upper_i <= bounds_i[j] || upper_i <= cdist[triindex(a, j)] {
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
                    })
                    .collect();
                let mut total = 0_usize;
                for (c, ds, dc) in deltas {
                    total += c;
                    for j in 0..k {
                        math::add_assign(sums.center_mut(j), &ds[j * d..j * d + d], d);
                        csize[j] = (csize[j] as i64 + dc[j]) as usize;
                    }
                }
                break 'iter total;
            }
            // serial path (no parallel feature, or n below threshold)
            let mut c = 0;
            for i in 0..n {
                let aa = assign[i];
                // Update bounds
                let bounds_i = &mut bounds[i * k..i * k + k];
                let mut upper_i = bounds_i[aa] + cmov[aa];
                math::sub_assign(bounds_i, &cmov, k); // we overwrite [aa] below!
                if upper_i < cnear[aa] {
                    bounds_i[aa] = upper_i; // store upper bound
                    continue;
                }
                // Check bounds
                let (mut loaded, mut upper_tight, mut a) = (false, false, aa);
                for j in 0..k {
                    if j == aa || upper_i <= bounds_i[j] || upper_i <= cdist[triindex(a, j)] {
                        continue;
                    }
                    // Make upper bound tight first:
                    if !upper_tight {
                        if !loaded {
                            data.load_into(i, &mut scratch, d);
                            loaded = true;
                        }
                        upper_i = math::sqdist(cent.center(aa), &scratch, d).sqrt();
                        bounds_i[aa] = upper_i;
                        upper_tight = true;
                        if upper_i <= bounds_i[j] || upper_i <= cdist[triindex(a, j)] {
                            continue;
                        }
                    }
                    // Make lower tight
                    bounds_i[j] = math::sqdist(cent.center(j), &scratch, d).sqrt();
                    if bounds_i[j] < upper_i {
                        a = j;
                        upper_i = bounds_i[j];
                    }
                }
                bounds_i[a] = upper_i; // store upper bound
                if a != aa {
                    assign[i] = a;
                    csize[aa] -= 1;
                    csize[a] += 1;
                    math::sub_assign(sums.center_mut(aa), &scratch, d);
                    math::add_assign(sums.center_mut(a), &scratch, d);
                    c += 1;
                }
            }
            c
        };
        if changed == 0 {
            break;
        }
    }
    // elkan does not compute inertia directly, return without it
    KMeansResult::without_inertia(cent.into_ndarray(), assign, iter)
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
        let result = elkan::<_, _, _>(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &result.centers, &result.assignments);
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
        assert_eq!(result.iterations, 11, "niter not as expected");
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let _res1 = elkan::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);
        let n1 = _res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let _res2 = elkan::<_, _, _>(&dataset, 5, &mut init2, 100, tol);
        let n2 = _res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
