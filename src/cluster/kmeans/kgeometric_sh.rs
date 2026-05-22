use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::kgeometric::weiszfeld_step;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Initial assignment for the simplified‑Hamerly variant of k‑geometric.
///
/// Compared with `kgeo_initial_assignment` this helper also computes the
/// two smallest distances per point so that we can maintain Hamerly bounds
/// during the main loop.  The distances are returned as *non‑squared* values
/// to keep the later code simple.  Additionally, if the initializer provides
/// squared distances (via `init_with_distances`), we return those as an
/// optional cache for the first Weiszfeld step.
///
/// A scratch buffer of length `d` is required for temporary storage during
/// distance calculations.
#[inline(always)]
fn kgeo_sh_initial_assignment<N, A, I>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>, scratch: &mut [N],
) -> (Vec<usize>, Vec<usize>, Vec<N>, Vec<N>)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0_usize; n];
    let mut csize = vec![0_usize; k];
    let mut lower = vec![N::infinity(); n];
    let prev_dist: Vec<N>;

    if init.uses_distances() {
        let mut cache = vec![N::infinity(); n];
        init.init_with_distances::<A, _>(
            data,
            cent,
            k,
            Some(
                #[inline(always)]
                |j: usize, i: usize, dist: N| {
                    if dist < cache[i] {
                        assign[i] = j;
                        lower[i] = cache[i];
                        cache[i] = dist;
                    } else if dist < lower[i] {
                        lower[i] = dist;
                    }
                },
            ),
        );
        for i in 0..n {
            let a = assign[i];
            lower[i] = lower[i].sqrt();
            cache[i] = cache[i].sqrt();
            csize[a] += 1;
        }
        prev_dist = cache;
    } else {
        init.init::<A>(data, cent, k);
        let mut cache = vec![N::infinity(); n];
        for i in 0..n {
            data.load_into(i, scratch, d);
            let (mut a, mut s, mut s2) = (k, N::infinity(), N::infinity());
            for j in 0..k {
                let tmp = math::sqdist(cent.center(j), scratch, d);
                if tmp < s {
                    (a, s, s2) = (j, tmp, s);
                } else if tmp < s2 {
                    s2 = tmp;
                }
            }
            csize[a] += 1;
            assign[i] = a;
            lower[i] = s2.sqrt();
            cache[i] = s.sqrt();
        }
        prev_dist = cache;
    }
    (assign, csize, lower, prev_dist)
}

#[inline(always)]
fn kgeo_sh_exact_reassignment<N, A>(
    data: &A, k: usize, cent: &Centers<N>, scratch: &mut [N],
) -> (Vec<usize>, Vec<usize>, Vec<N>, N)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0_usize; n];
    let mut csize = vec![0_usize; k];
    let mut lower = vec![N::infinity(); n];
    let mut sum = N::zero();

    for i in 0..n {
        data.load_into(i, scratch, d);
        let (mut a, mut s_sq, mut s2_sq) =
            (0, math::sqdist(cent.center(0), scratch, d), N::infinity());
        for j in 1..k {
            let tmp_sq = math::sqdist(cent.center(j), scratch, d);
            if tmp_sq < s_sq {
                (a, s_sq, s2_sq) = (j, tmp_sq, s_sq);
            } else if tmp_sq < s2_sq {
                s2_sq = tmp_sq;
            }
        }
        let upper = s_sq.sqrt();
        assign[i] = a;
        csize[a] += 1;
        lower[i] = s2_sq.sqrt();
        sum += upper;
    }

    (assign, csize, lower, sum)
}

/// Simplified‑Hamerly k‑geometric median clustering.
#[inline(always)]
pub fn kgeometric_sh<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N, steps: usize,
) -> Result<KMeansResult<N>, String>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N> + Sync,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);

    let (mut assign, mut csize, mut lower, mut prev_dist) =
        kgeo_sh_initial_assignment::<N, A, I>(data, k, init, &mut cent, &mut scratch);

    // compute initial per‑point log‑likelihood sum for compatibility with
    // the original kgeometric implementation
    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };

        // compute new centers via Weiszfeld steps and track movements
        let (mut most, mut cmov1, mut cmov2) = (0, N::zero(), N::zero());
        let mut diff_sq = N::zero();

        let mut current = vec![N::zero(); d];
        for (j, &count) in csize.iter().enumerate() {
            if count > 0 {
                // start from current center
                math::copy(&mut current, cent.center(j), d);
                for step in 0..steps {
                    let updated = weiszfeld_step::<N, A>(
                        data,
                        j,
                        &assign,
                        &current,
                        if step == 0 { Some(prev_dist.as_slice()) } else { None },
                        false,
                    );
                    math::copy(&mut current, &updated, d);
                }
                let tmp = math::sqdist(&current, cent.center(j), d).sqrt();
                if tol > N::zero() {
                    diff_sq += tmp * tmp;
                }
                math::copy(cent.center_mut(j), &current, d);
                if tmp > cmov1 {
                    (most, cmov1, cmov2) = (j, tmp, cmov1);
                } else if tmp > cmov2 {
                    cmov2 = tmp;
                }
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

        // Semi-Hamerly reassignment: the distance to the assigned center is
        // always kept exact (required for Weiszfeld); only a lower bound on
        // the second-closest center is used to skip the full comparison.
        let changed = 'iter: {
            #[cfg(feature = "parallel")]
            if n >= crate::math::PARALLEL_ROW_THRESHOLD {
                use rayon::prelude::*;
                let chunk_size = n.div_ceil(rayon::current_num_threads());
                let deltas: Vec<(usize, Vec<i64>)> = assign
                    .par_chunks_mut(chunk_size)
                    .zip(lower.par_chunks_mut(chunk_size))
                    .zip(prev_dist.par_chunks_mut(chunk_size))
                    .enumerate()
                    .map(|(ti, ((assign_chunk, lower_chunk), prev_dist_chunk))| {
                        let i0 = ti * chunk_size;
                        let chunk_n = assign_chunk.len();
                        let mut scratch = vec![N::zero(); d];
                        let mut delta_csize = vec![0i64; k];
                        let mut local_changed = 0usize;
                        for ci in 0..chunk_n {
                            let i = i0 + ci;
                            let aa = assign_chunk[ci];
                            // lower bound on the distance to the second-closest center
                            let lower_i = lower_chunk[ci] - if aa != most { cmov1 } else { cmov2 };
                            data.load_into(i, &mut scratch, d);
                            // always compute exact distance to the currently assigned center
                            let actual = math::sqdist(cent.center(aa), &scratch, d).sqrt();
                            prev_dist_chunk[ci] = actual;
                            // If actual <= lower bound, assignment cannot change.
                            if actual <= lower_i {
                                lower_chunk[ci] = lower_i;
                                continue;
                            }
                            let (mut a, mut s, mut b, mut s2) = (aa, actual, k, N::infinity());
                            for j in 0..k {
                                if j != aa {
                                    let tmp = math::sqdist(cent.center(j), &scratch, d).sqrt();
                                    if tmp < s {
                                        (a, s, b, s2) = (j, tmp, a, s);
                                    } else if tmp < s2 {
                                        (b, s2) = (j, tmp);
                                    }
                                }
                            }
                            lower_chunk[ci] = if b == aa { actual } else { s2 };
                            if a != aa {
                                assign_chunk[ci] = a;
                                delta_csize[aa] -= 1;
                                delta_csize[a] += 1;
                                local_changed += 1;
                            }
                            prev_dist_chunk[ci] = s;
                        }
                        (local_changed, delta_csize)
                    })
                    .collect();
                let mut total = 0usize;
                for (c, dc) in deltas {
                    total += c;
                    for j in 0..k {
                        csize[j] = (csize[j] as i64 + dc[j]) as usize;
                    }
                }
                break 'iter total;
            }
            // serial path (no parallel feature, or n below threshold)
            let mut c = 0;
            for i in 0..n {
                let aa = assign[i];
                // lower bound on the distance to the second-closest center
                let lower_i = lower[i] - if aa != most { cmov1 } else { cmov2 };

                data.load_into(i, &mut scratch, d);
                // always compute the exact distance to the currently assigned center
                let actual = math::sqdist(cent.center(aa), &scratch, d).sqrt();
                prev_dist[i] = actual;

                // If the exact distance to the assigned center is at most the lower
                // bound on the second-closest center, the assignment cannot change.
                if actual <= lower_i {
                    lower[i] = lower_i;
                    continue;
                }

                let (mut a, mut s, mut b, mut s2) = (aa, actual, k, N::infinity());
                for j in 0..k {
                    if j != aa {
                        let tmp = math::sqdist(cent.center(j), &scratch, d).sqrt();
                        if tmp < s {
                            (a, s, b, s2) = (j, tmp, a, s);
                        } else if tmp < s2 {
                            (b, s2) = (j, tmp);
                        }
                    }
                }
                lower[i] = if b == aa { actual } else { s2 };
                if a != aa {
                    assign[i] = a;
                    csize[aa] -= 1;
                    csize[a] += 1;
                    c += 1;
                }
                prev_dist[i] = s;
            }
            c
        };
        if changed == 0 {
            break;
        }
    }

    // Final exact pass so returned assignments, bounds, and distances all
    // correspond to the final centers.
    let (assign, _final_csize, _final_lower, lastsum) =
        kgeo_sh_exact_reassignment::<N, A>(data, k, &cent, &mut scratch);

    Ok(KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum))
}

/// Public entry point with backend dispatch just like the original kgeometric
// basic smoke tests for the new algorithm

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn sh_basic_matches_plain() {
        let mat = gen_test_data((50, 2), Pcg32::seed_from_u64(123));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(1));
        let res1 =
            crate::cluster::kmeans::kgeometric::kgeometric(&dataset, 5, &mut init1, 50, 1e-4, 1)
                .unwrap();
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(1));
        let res2 = kgeometric_sh(&dataset, 5, &mut init2, 50, 1e-4, 1).unwrap();
        // assignments may differ because of numerical rounding; recompute
        // Euclidean geometric loss from the returned centers/assignments
        // rather than trusting the `inertia` field which may be under‑
        // counted when Hamerly bounds avoid explicit distance computations.
        fn euclidean_loss<A>(data: &A, centers: &Array2<f64>, assign: &[usize]) -> f64
        where
            A: Dataset<f64>,
        {
            let (n, d) = (data.nrows(), data.ncols());
            let mut scratch = vec![0.0_f64; d];
            let mut loss = 0.0;
            for (i, &idx) in assign.iter().enumerate().take(n) {
                data.load_into(i, &mut scratch, d);
                let row = centers.row(idx);
                let sq = math::sqdist(&scratch, row.as_slice().unwrap(), d);
                loss += sq.sqrt();
            }
            loss
        }
        let loss1 = euclidean_loss(&dataset, &res1.centers, &res1.assignments);
        let loss2 = euclidean_loss(&dataset, &res2.centers, &res2.assignments);
        assert!((loss1 - loss2).abs() < 1e-6, "losses differ: {} vs {}", loss1, loss2);
    }
}
