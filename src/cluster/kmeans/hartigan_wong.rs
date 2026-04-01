use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Hartigan-Wong k-means (Algorithm AS 136).
///
/// This method alternates between an optimal-transfer stage and a
/// quick-transfer stage, mirroring the original AS 136 algorithm.
#[inline(always)]
pub fn hartigan_wong<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);

    // Initial assignment (IC1) and center computation.
    let (mut assign, mut csize, _) = crate::cluster::kmeans::lloyd::lloyd_initial_assignment::<
        N,
        A,
        I,
    >(data, k, init, &mut cent, &mut sums, &mut scratch);

    // Data structures needed by AS 136.
    let mut second = vec![0_usize; n]; // IC2
    let mut dist1 = vec![N::zero(); n]; // D(i) = dist(i, IC1(i)) * AN1(IC1(i))
    let mut dist2 = vec![N::zero(); n]; // unused except as temp
    let mut an1 = vec![N::zero(); k];
    let mut an2 = vec![N::zero(); k];
    let mut ncp = vec![-1_isize; k];
    let mut itrans = vec![1_u8; k];
    let mut live = vec![0_usize; k];

    let big = N::from(1e30).unwrap_or(N::infinity());

    fn update_an<N>(csize: &[usize], an1: &mut [N], an2: &mut [N], l: usize, big: N)
    where
        N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    {
        let sz = csize[l];
        let sz_n: N = N::from(sz).unwrap();
        an2[l] = sz_n / (sz_n + N::one());
        an1[l] = if sz > 1 { sz_n / (sz_n - N::one()) } else { big };
    }

    for l in 0..k {
        update_an(&csize, &mut an1, &mut an2, l, big);
        live[l] = n + 1;
    }

    // Initial computation of IC2, dist1 and dist2.
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        let l1 = assign[i];
        let mut best = l1;
        let mut best_d = N::infinity();
        let d1 = math::sqdist(cent.center(l1), &scratch, d);
        dist1[i] = d1 * an1[l1];
        for l in 0..k {
            if l == l1 {
                continue;
            }
            let d = math::sqdist(cent.center(l), &scratch, d);
            if d < best_d {
                best_d = d;
                best = l;
            }
        }
        second[i] = best;
        dist2[i] = best_d;
    }

    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
        let old_cent = if tol > N::zero() { Some(cent.clone()) } else { None };

        // --- Optimal transfer stage ------------------------------------------------
        // Make clusters that were updated in the quick-transfer stage "live".
        for l in 0..k {
            if itrans[l] == 1 {
                live[l] = n + 1;
            }
        }

        let mut indx = 0_usize;
        for i in 0..n {
            indx += 1;
            let l1 = assign[i];

            if csize[l1] == 1 {
                continue;
            }

            if ncp[l1] != 0 {
                data.load_into(i, &mut scratch, d);
                let d1 = math::sqdist(cent.center(l1), &scratch, d);
                dist1[i] = d1 * an1[l1];
            }

            // Determine the best candidate cluster (starting from IC2).
            let mut l2 = second[i];
            data.load_into(i, &mut scratch, d);
            let mut r2 = math::sqdist(cent.center(l2), &scratch, d) * an2[l2];

            // Search for a better transfer.
            for l in 0..k {
                if l == l1 || l == l2 {
                    continue;
                }
                if i >= live[l1] && i >= live[l] {
                    continue;
                }
                let r2cand = math::sqdist(cent.center(l), &scratch, d) * an2[l];
                if r2cand < r2 {
                    r2 = r2cand;
                    l2 = l;
                }
            }

            if r2 < dist1[i] {
                // Perform transfer from l1 to l2.
                indx = 0;
                let mut center_buf = vec![N::zero(); d];

                // remove from l1
                csize[l1] -= 1;
                math::sub_assign(sums.center_mut(l1), &scratch, d);
                if csize[l1] > 0 {
                    let recip = N::from(csize[l1]).unwrap().recip();
                    math::mul(&mut center_buf, sums.center(l1), recip, d);
                    math::copy(cent.center_mut(l1), &center_buf, d);
                }

                // add to l2
                csize[l2] += 1;
                math::add_assign(sums.center_mut(l2), &scratch, d);
                let recip = N::from(csize[l2]).unwrap().recip();
                math::mul(&mut center_buf, sums.center(l2), recip, d);
                math::copy(cent.center_mut(l2), &center_buf, d);

                assign[i] = l2;
                second[i] = l1;

                ncp[l1] = (i + 1) as isize;
                ncp[l2] = (i + 1) as isize;
                itrans[l1] = 1;
                itrans[l2] = 1;

                update_an(&csize, &mut an1, &mut an2, l1, big);
                update_an(&csize, &mut an1, &mut an2, l2, big);

                dist1[i] = r2 * N::one();
            }

            if indx == n {
                break;
            }
        }

        if indx == n {
            break;
        }

        // --- Quick transfer stage --------------------------------------------------
        let mut qstep = 0_usize;
        let max_qsteps = maxiter;
        let mut qchanged = true;

        while qchanged && qstep < max_qsteps {
            qstep += 1;
            qchanged = false;

            for i in 0..n {
                let l1 = assign[i];
                if csize[l1] == 1 {
                    continue;
                }

                let l2 = second[i];
                if (qstep <= (ncp[l1] as usize)) && (qstep <= (ncp[l2] as usize)) {
                    continue;
                }

                data.load_into(i, &mut scratch, d);
                let d2 = math::sqdist(cent.center(l2), &scratch, d);

                let r2 = dist1[i] / an2[l2];
                if d2 >= r2 {
                    continue;
                }

                // Transfer l1 -> l2.
                qchanged = true;
                let mut center_buf = vec![N::zero(); d];

                csize[l1] -= 1;
                math::sub_assign(sums.center_mut(l1), &scratch, d);
                if csize[l1] > 0 {
                    let recip = N::from(csize[l1]).unwrap().recip();
                    math::mul(&mut center_buf, sums.center(l1), recip, d);
                    math::copy(cent.center_mut(l1), &center_buf, d);
                }

                csize[l2] += 1;
                math::add_assign(sums.center_mut(l2), &scratch, d);
                let recip = N::from(csize[l2]).unwrap().recip();
                math::mul(&mut center_buf, sums.center(l2), recip, d);
                math::copy(cent.center_mut(l2), &center_buf, d);

                assign[i] = l2;
                second[i] = l1;

                ncp[l1] = (qstep + n) as isize;
                ncp[l2] = (qstep + n) as isize;
                itrans[l1] = 1;
                itrans[l2] = 1;

                update_an(&csize, &mut an1, &mut an2, l1, big);
                update_an(&csize, &mut an1, &mut an2, l2, big);

                dist1[i] = d2 * an1[l2];
            }
        }

        if k == 2 {
            break;
        }

        for l in 0..k {
            ncp[l] = 0;
        }

        // tolerance check
        if let Some(old) = old_cent {
            let diff = cent.diff_frobenius_norm(&old);
            let norm = old.frobenius_norm();
            let rel = if norm == N::zero() { diff } else { diff / norm };
            if rel <= tol {
                break;
            }
        }
    }

    // Compute exact inertia for the final assignment.
    let mut sum = N::zero();
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        sum += math::sqdist(cent.center(assign[i]), &scratch, d);
    }

    KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, sum)
}

/// Hartigan-Wong k-means (Algorithm AS 136) with quick-transfer heuristic.
///
/// This variant attempts to avoid the full scan for every point by first
/// testing the second-closest centroid (quick transfer). Only if that transfer
/// is not beneficial does it fall back to the full optimal transfer scan.
#[inline(always)]
pub fn hartigan_wong_quick<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);
    let (mut assign, mut csize, _) = crate::cluster::kmeans::lloyd::lloyd_initial_assignment::<
        N,
        A,
        I,
    >(data, k, init, &mut cent, &mut sums, &mut scratch);

    let mut best2_idx = vec![0_usize; n];
    let mut best2_dist = vec![N::infinity(); n];

    // Precompute nearest-other-centroid for each point.
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        let r = assign[i];
        let mut best_j = r;
        let mut best_d = N::infinity();
        for j in 0..k {
            if j == r {
                continue;
            }
            let ds = math::sqdist(cent.center(j), &scratch, d);
            if ds < best_d {
                best_d = ds;
                best_j = j;
            }
        }
        best2_idx[i] = best_j;
        best2_dist[i] = best_d;
    }

    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
        let old_cent = if tol > N::zero() { Some(cent.clone()) } else { None };
        let mut changed = 0;

        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            let r = assign[i];
            let nr = csize[r];
            if nr <= 1 {
                continue;
            }
            let dr = math::sqdist(cent.center(r), &scratch, d);

            // Try quick transfer using the most promising alternative cluster.
            let mut best_j = best2_idx[i];
            let ds = math::sqdist(cent.center(best_j), &scratch, d);
            best2_dist[i] = ds;

            let ns = csize[best_j];
            let best_delta = dr * N::from(nr).unwrap() / (N::from(nr).unwrap() - N::one())
                - ds * N::from(ns).unwrap() / (N::from(ns).unwrap() + N::one());

            let moved = if best_delta > N::zero() {
                // Quick transfer succeeds.
                true
            } else {
                // Fallback: perform full optimal transfer scan.
                let mut best_delta_opt = best_delta;
                for j in 0..k {
                    if j == r || j == best_j {
                        continue;
                    }
                    let ns = csize[j];
                    let ds = math::sqdist(cent.center(j), &scratch, d);
                    let delta = dr * N::from(nr).unwrap() / (N::from(nr).unwrap() - N::one())
                        - ds * N::from(ns).unwrap() / (N::from(ns).unwrap() + N::one());
                    if delta > best_delta_opt {
                        best_delta_opt = delta;
                        best_j = j;
                    }
                }
                best_j != r && best_delta_opt > N::zero()
            };

            if moved {
                // perform move
                csize[r] -= 1;
                math::sub_assign(sums.center_mut(r), &scratch, d);
                if csize[r] > 0 {
                    let recip = N::from(csize[r]).unwrap().recip();
                    math::mul(&mut scratch, sums.center(r), recip, d);
                    math::copy(cent.center_mut(r), &scratch, d);
                } else {
                    for v in cent.center_mut(r).iter_mut() {
                        *v = N::zero();
                    }
                    for v in sums.center_mut(r).iter_mut() {
                        *v = N::zero();
                    }
                }

                csize[best_j] += 1;
                math::add_assign(sums.center_mut(best_j), &scratch, d);
                let recip = N::from(csize[best_j]).unwrap().recip();
                math::mul(&mut scratch, sums.center(best_j), recip, d);
                math::copy(cent.center_mut(best_j), &scratch, d);

                assign[i] = best_j;
                changed += 1;

                // Recompute best2 for this point after move.
                data.load_into(i, &mut scratch, d);
                let r_new = best_j;
                let mut best_j2 = r_new;
                let mut best_d2 = N::infinity();
                for j in 0..k {
                    if j == r_new {
                        continue;
                    }
                    let ds2 = math::sqdist(cent.center(j), &scratch, d);
                    if ds2 < best_d2 {
                        best_d2 = ds2;
                        best_j2 = j;
                    }
                }
                best2_idx[i] = best_j2;
                best2_dist[i] = best_d2;
            }
        }

        // tolerance check
        if let Some(old) = old_cent {
            let diff = cent.diff_frobenius_norm(&old);
            let norm = old.frobenius_norm();
            let rel = if norm == N::zero() { diff } else { diff / norm };
            if rel <= tol {
                break;
            }
        }

        if changed == 0 {
            break;
        }
    }

    // Compute exact inertia for the final assignment.
    let mut sum = N::zero();
    for i in 0..n {
        data.load_into(i, &mut scratch, d);
        sum += math::sqdist(cent.center(assign[i]), &scratch, d);
    }

    KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, sum)
}

/// Hartigan-Wong k-means (Algorithm AS 136).

/// Hartigan-Wong k-means with quick transfer heuristic (fast path when the
/// second-closest centroid yields improvement).

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::cluster::kmeans::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(42)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res = hartigan_wong(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert!(res.inertia.is_some() && (loss - res.inertia.unwrap()).abs() < 1e-12);
        assert!(res.iterations <= 100);
    }

    #[test]
    fn test_quick() {
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(42)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res = hartigan_wong_quick(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert!(res.inertia.is_some() && (loss - res.inertia.unwrap()).abs() < 1e-12);
        assert!(res.iterations <= 100);
    }

    #[test]
    fn test_hartigan_wong_can_escape_lloyd_tie_assignment() {
        // Construct a simple unbalanced dataset where Lloyd's algorithm gets stuck
        // because the "bridge" point is exactly equidistant to both initial
        // centroids and therefore stays assigned to the large cluster.
        // Hartigan-Wong can transfer that point to the small cluster because it
        // reduces the overall WCSS.

        // Large cluster (20 points at (0,0)), tiny cluster (1 point at (10,0)),
        // and a bridge point at (5,0).
        let mut points = Vec::new();
        for _ in 0..20 {
            points.push(0.0);
            points.push(0.0);
        }
        points.push(10.0);
        points.push(0.0);
        // Bridge point is slightly closer to the large cluster center at 0.
        // Hartigan-Wong should still move it to the small cluster if that reduces WCSS.
        points.push(4.9);
        points.push(0.0);

        let mat = ndarray::Array2::from_shape_vec((22, 2), points).unwrap();
        let dataset = NdArrayDataset::new(&mat);

        struct FixedInit {
            centers: Vec<[f64; 2]>,
        }

        impl<N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum> Initialization<N> for FixedInit {
            fn uses_distances(&self) -> bool { false }

            fn init<A>(&mut self, _data: &A, cent: &mut Centers<N>, k: usize)
            where
                A: Dataset<N>,
            {
                assert_eq!(k, self.centers.len());
                for (i, c) in self.centers.iter().enumerate().take(k) {
                    let mut buf = vec![N::zero(); 2];
                    buf[0] = N::from(c[0]).unwrap();
                    buf[1] = N::from(c[1]).unwrap();
                    math::copy(cent.center_mut(i), &buf, 2);
                }
            }

            fn init_with_distances<A, F>(
                &mut self, data: &A, cent: &mut Centers<N>, k: usize, _callback: Option<F>,
            ) where
                A: Dataset<N>,
                F: FnMut(usize, usize, N),
            {
                self.init::<A>(data, cent, k);
            }
        }

        let mut init = FixedInit { centers: vec![[0.0, 0.0], [10.0, 0.0]] };
        let mut init2 = FixedInit { centers: vec![[0.0, 0.0], [10.0, 0.0]] };

        let res_lloyd = crate::cluster::kmeans::lloyd::lloyd(&dataset, 2, &mut init, 100, 0.0);
        let res_hw = hartigan_wong(&dataset, 2, &mut init2, 100, 0.0);

        let _loss_lloyd = compute_loss(&dataset, &res_lloyd.centers, &res_lloyd.assignments);
        let _loss_hw = compute_loss(&dataset, &res_hw.centers, &res_hw.assignments);

        assert!(
            _loss_hw + 1e-12 < _loss_lloyd,
            "Hartigan-Wong should find a better clustering than Lloyd in this setup"
        );
    }
}
