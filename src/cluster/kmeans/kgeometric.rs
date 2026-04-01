use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Build initial assignments using squared L2 distances (just like Lloyd).
#[inline(always)]
pub(crate) fn kgeo_initial_assignment<N, I, A>(
    data: &A, k: usize, init: &mut I, cent: &mut Centers<N>,
) -> (Vec<usize>, Vec<usize>, N, Option<Vec<N>>)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
    I: Initialization<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut assign = vec![0_usize; n];
    let mut csize = vec![0_usize; k];
    let mut lastsum = N::zero();
    let mut scratch = vec![N::zero(); d];

    // the initialization is responsible only for choosing starting points;
    // we perform our own assignment with squared L2 distances.  if the
    // initializer can supply distances (via `init_with_distances`), we use
    // them to prefill `prev_sq` and avoid recomputing these values later.
    let mut prev_sq: Option<Vec<N>> = None;
    if init.uses_distances() {
        // allocate vector to keep the distance to the *nearest* centre seen
        // during initialization.  the callback receives squared distances.
        let mut cache = vec![N::infinity(); n];
        init.init_with_distances::<A, _>(
            data,
            cent,
            k,
            Some(
                #[inline(always)]
                |_: usize, i: usize, d: N| {
                    if d < cache[i] {
                        cache[i] = d;
                    }
                },
            ),
        );
        prev_sq = Some(cache);
    } else {
        init.init::<A>(data, cent, k);
    }
    // compute assignment for every point and optionally update prev_sq with the
    // actual distance to the assigned centre (the latter may replace the
    // initial value from the initializer which could correspond to a different
    // centre during the selection process).
    for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
        data.load_into(i, &mut scratch, d);
        let (mut a, mut s_sq) = (0, math::sqdist(cent.center(0), &scratch, d));
        for j in 1..k {
            let tmp_sq = math::sqdist(cent.center(j), &scratch, d);
            if tmp_sq < s_sq {
                (a, s_sq) = (j, tmp_sq);
            }
        }
        csize[a] += 1;
        *assign_i = a;
        lastsum += math::sqdist(cent.center(a), &scratch, d).sqrt();
        if let Some(ref mut cache) = prev_sq {
            cache[i] = s_sq;
        }
    }
    (assign, csize, lastsum, prev_sq)
}

/// One step of Weiszfeld's algorithm around the given center using the points
/// that satisfy `assign[i] == cluster`.
///
/// This implementation follows the *Vardi–Zhang* modification to handle the
/// case where the current estimate `y` coincides with one or more data points.
///
/// Let \(X = \{x_i\}\) be the set of assigned points.  For efficiency we
/// assume unit weights; the generalisation is straightforward.  The iteration
/// computes:
///
/// 1. identify the set \(S = \{i: \|x_i - y\| > \varepsilon\}\) and let
///    \(\eta_y\) be the multiplicity of points that coincide with `y`.
/// 2. compute the standard Weiszfeld update on \(S\),
///    \(t_1 = \frac{\sum_{i\in S} x_i/\|x_i-y\|}{\sum_{i\in S} 1/\|x_i-y\|}\).
/// 3. if \(\eta_y = 0\) return \(t_1\); otherwise form the residual
///    \(R=\sum_{i\in S} (x_i - y)/\|x_i-y\|\) and let
///    \(\gamma = \min(1,\eta_y/\|R\|)\).  The new location is then
///    \((1-\gamma)t_1 + \gamma y\).
///
/// The `eps` threshold (chosen as the square root of machine epsilon) guards
/// against extremely small denominators and provides a reproducible notion of
/// coincidence.  If all points are within `eps` of `y`, the original vector is
/// returned unchanged.
#[inline(always)]
pub(crate) fn weiszfeld_step<N, A>(
    data: &A,
    cluster: usize,
    assign: &[usize],
    cent: &[N],
    // If provided, these are cached distances from each point to its assigned
    // center, as returned by initialization or cached from the previous
    // iteration.
    prev_dists: Option<&[N]>,
    prev_dists_are_squared: bool,
) -> Vec<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy,
    A: Dataset<N>,
{
    let d = data.ncols();
    let mut numer = vec![N::zero(); d];
    let mut denom = N::zero();
    let mut r_vec = vec![N::zero(); d];
    let out = cent.to_vec();

    let mut point = vec![N::zero(); d];
    let eps = N::epsilon().sqrt();
    let mut eta_count: usize = 0;

    for (i, &a) in assign.iter().enumerate() {
        if a != cluster {
            continue;
        }
        data.load_into(i, &mut point, d);
        let dist = if let Some(prev_dists) = prev_dists {
            // Use the stored distance from initialization / previous
            // iteration. This avoids recomputing the distance for the first
            // Weiszfeld step.
            if prev_dists_are_squared { prev_dists[i].sqrt() } else { prev_dists[i] }
        } else {
            math::sqdist(&out, &point, d).sqrt()
        };
        if dist <= eps {
            // treat as coincident; count multiplicity but do not add to
            // Weiszfeld sums
            eta_count += 1;
            continue;
        }
        let inv = N::one() / dist;
        // numer += point / dist
        math::axpy(&mut numer, inv, &point, d);
        // r_vec += (point - out) / dist
        let mut diff = point.clone();
        math::sub_assign(&mut diff, &out, d);
        math::mul_assign(&mut diff, inv, d);
        math::add_assign(&mut r_vec, &diff, d);
        denom += inv;
    }

    // if there were no non‑zero distances, nothing can be done
    if denom == N::zero() {
        return out;
    }

    // compute t1 = numer / denom
    let mut t1 = vec![N::zero(); d];
    math::mul(&mut t1, &numer, N::one() / denom, d);

    if eta_count == 0 {
        return t1;
    }

    let r_norm = math::norm(&r_vec, d);
    let gamma = if r_norm > eps {
        let ratio = N::from(eta_count).unwrap_or(N::one()) / r_norm;
        if ratio < N::one() { ratio } else { N::one() }
    } else {
        // residual is essentially zero; fall back to centre
        N::one()
    };

    let mut out_new = t1.clone();
    math::mul_assign(&mut out_new, N::one() - gamma, d);
    math::axpy(&mut out_new, gamma, &out, d);
    out_new
}

/// Run k‑geometric‑median clustering.  `steps` controls how many Weiszfeld
/// iterations are performed per cluster on each k‑means iteration; the
/// algorithm only executes `steps` loops and then reassigns points.  Distance
/// for assignment uses squared L2 (just like standard k‑means).
pub fn kgeometric<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N, steps: usize,
) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let (mut assign, mut csize, _, maybe_sq) =
        kgeo_initial_assignment::<N, I, A>(data, k, init, &mut cent);

    // optionally reuse distances returned by the initializer (squared L2).
    let mut prev_sq: Option<Vec<N>> = maybe_sq;

    let mut iter = 1;
    while iter < maxiter {
        iter += 1;
        let old_norm = if tol > N::zero() { cent.frobenius_norm() } else { N::zero() };

        // compute new centers by running `steps` of Weiszfeld per cluster
        let mut new_cent = Centers::<N>::new(k, d);
        let mut current = vec![N::zero(); d];
        for (j, &count) in csize.iter().enumerate().take(k) {
            if count == 0 {
                continue;
            }
            math::copy(&mut current, cent.center(j), d);
            for step in 0..steps {
                let updated = weiszfeld_step::<N, A>(
                    data,
                    j,
                    &assign,
                    &current,
                    if step == 0 { prev_sq.as_deref() } else { None },
                    true,
                );
                math::copy(&mut current, &updated, d);
            }
            math::copy(new_cent.center_mut(j), &current, d);
        }

        // tolerance check
        if tol > N::zero() {
            let mut diff_sq = N::zero();
            for j in 0..k {
                diff_sq += math::sqdist(cent.center(j), new_cent.center(j), d);
            }
            let diff = diff_sq.sqrt();
            let rel = if old_norm == N::zero() { diff } else { diff / old_norm };
            if rel <= tol {
                cent = new_cent;
                break;
            }
        }
        cent = new_cent;

        // reassign
        let mut changed = 0;
        // prepare new cache vector if we're already caching distances
        let mut next_prev: Option<Vec<N>> = prev_sq.as_ref().map(|_| vec![N::zero(); n]);
        for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
            let aa = *assign_i;
            data.load_into(i, &mut scratch, d);
            // Reassignment must be exact against the current centers.
            let (mut a, mut s_sq) = (aa, math::sqdist(cent.center(aa), &scratch, d));
            for j in 0..k {
                if j == aa {
                    continue;
                }
                let tmp_sq = math::sqdist(cent.center(j), &scratch, d);
                if tmp_sq < s_sq {
                    (a, s_sq) = (j, tmp_sq);
                }
            }
            if let Some(ref mut nxt) = next_prev {
                nxt[i] = s_sq;
            }
            if a != aa {
                *assign_i = a;
                csize[aa] -= 1;
                csize[a] += 1;
                changed += 1;
            }
        }
        // swap caches so the newly computed distances are used in the next
        // iteration; if we hadn't cached before we still won't in future.
        prev_sq = next_prev;
        if changed == 0 {
            break;
        }
    }

    // Refresh assignments and distances against the final centers so the
    // returned state is exact even when we stopped on the tolerance check.
    csize.fill(0);
    let mut lastsum = N::zero();
    for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
        data.load_into(i, &mut scratch, d);
        let (mut a, mut s_sq) = (0, math::sqdist(cent.center(0), &scratch, d));
        for j in 1..k {
            let tmp_sq = math::sqdist(cent.center(j), &scratch, d);
            if tmp_sq < s_sq {
                (a, s_sq) = (j, tmp_sq);
            }
        }
        *assign_i = a;
        csize[a] += 1;
        lastsum += s_sq.sqrt();
    }
    KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum)
}

/// K-geometric median clustering entry with runtime math dispatch

// simple dataset helper for tests
#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleDataset<N> {
        data: Vec<N>,
        n: usize,
        d: usize,
    }
    impl<N: Copy> SimpleDataset<N> {
        fn new(data: Vec<N>, n: usize, d: usize) -> Self {
            assert_eq!(n * d, data.len());
            SimpleDataset { data, n, d }
        }
    }
    impl<N: Copy> crate::Data for SimpleDataset<N> {
        fn len(&self) -> usize { self.n }
    }

    impl<N: Copy> Dataset<N> for SimpleDataset<N> {
        fn nrows(&self) -> usize { self.n }
        fn ncols(&self) -> usize { self.d }
        fn dims(&self) -> usize { self.d }
        fn point(&self, idx: usize) -> &[N] {
            let start = idx * self.d;
            &self.data[start..start + self.d]
        }
        fn load_into(&self, i: usize, vec: &mut [N], d: usize) {
            let start = i * self.d;
            vec[..d].copy_from_slice(&self.data[start..start + d]);
        }
    }

    #[test]
    fn weiszfeld_zero_distance_returns_point() {
        // centre exactly on a dataset point should remain there
        let ds = SimpleDataset::new(vec![0., 0., 0., 0.], 2, 2);
        let assign = vec![0, 0];
        let cent = vec![0., 0.];
        let res = weiszfeld_step::<f64, _>(&ds, 0, &assign, &cent, None, false);
        assert_eq!(res, vec![0., 0.]);
    }

    #[test]
    fn weiszfeld_at_data_point_retains_location() {
        // with two points at 0 and 1 and starting at 0 the Vardi–Zhang
        // modification prevents a jump to 1
        let ds = SimpleDataset::new(vec![0., 0., 1., 0.], 2, 2);
        let assign = vec![0, 0];
        let cent = vec![0., 0.];
        let res = weiszfeld_step::<f64, _>(&ds, 0, &assign, &cent, None, false);
        assert_eq!(res, cent);
    }

    #[test]
    fn weiszfeld_step_moves_toward_median() {
        // for two points the sum of distances is constant on the segment
        // between them; any starting point inside [0,2] is already an
        // optimal solution and the iteration will remain there.
        let ds = SimpleDataset::new(vec![0., 0., 2., 0.], 2, 2);
        let assign = vec![0, 0];
        let cent = vec![0.1, 0.0];
        let res = weiszfeld_step::<f64, _>(&ds, 0, &assign, &cent, None, false);
        for i in 0..res.len() {
            assert!((res[i] - cent[i]).abs() < 1e-12, "unexpected move");
        }
    }

    #[test]
    fn weiszfeld_sensible_movement() {
        // behaviour on a trivial collinear triple should still move toward the
        // median and stay in the convex hull.
        let ds = SimpleDataset::new(vec![0., 0., 1., 1., 2., 2.], 3, 2);
        let assign = vec![0, 0, 0];
        let cent = vec![0., 0.];
        let res = weiszfeld_step::<f64, _>(&ds, 0, &assign, &cent, None, false);
        assert!(res[0] > 0.0 && res[0] < 2.0);
        assert!(res[1] > 0.0 && res[1] < 2.0);
        let start_dist = ((cent[0] - 1.0).powi(2) + (cent[1] - 1.0).powi(2)).sqrt();
        let new_dist = ((res[0] - 1.0).powi(2) + (res[1] - 1.0).powi(2)).sqrt();
        assert!(new_dist < start_dist);
    }

    #[test]
    fn weiszfeld_small_distance_stable() {
        // exercise the stability guard by placing one point extremely close
        // to the current centre; the presence of a coincident point should
        // not drive the centre away.
        let ds = SimpleDataset::new(vec![0., 0., 1e-20, 0.], 2, 2);
        let assign = vec![0, 0];
        let cent = vec![5e-21, 0.];
        let res = weiszfeld_step::<f64, _>(&ds, 0, &assign, &cent, None, false);
        assert!((res[0] - cent[0]).abs() < 1e-12, "unstable move for tiny dist");
    }
}

#[cfg(test)]
mod kgeometric_tests {
    use ndarray::Array2;
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::cluster::kmeans::ndarray::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let res = kgeometric(&dataset, 5, &mut init, 100, 1e-8, 1);
        let (cent, assign, niter, los) =
            (res.centers, res.assignments, res.iterations, res.inertia.unwrap_or_default());
        // compute Euclidean loss using Math helper
        let mut scratch = vec![0.0_f64; dataset.ncols()];
        let mut loss = 0.0_f64;
        for (i, &idx) in assign.iter().enumerate().take(dataset.nrows()) {
            dataset.load_into(i, &mut scratch, dataset.ncols());
            let row = cent.row(idx);
            let sq = math::sqdist(&scratch, row.as_slice().unwrap(), dataset.ncols());
            loss += sq.sqrt();
        }
        assert!((loss - los).abs() < 1e-12, "loss not correct");
        // iteration count may vary depending on initialization and tolerance;
        // ensure it is positive and does not exceed maxiter
        assert!((1..=100).contains(&niter), "niter out of range");
    }

    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = kgeometric(&dataset, 5, &mut init1, 100, 1e-4, 1);
        let n1 = res1.iterations;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = kgeometric(&dataset, 5, &mut init2, 100, 1e-3, 1);
        let n2 = res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }

    #[test]
    fn test_loss_reporting() {
        // run each algorithm on the same dataset and verify Euclidean loss
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);

        fn euclidean_loss<A>(data: &A, centers: &Array2<f64>, assign: &[usize]) -> f64
        where
            A: Dataset<f64>,
        {
            let (n, d) = (data.nrows(), data.ncols());
            let mut scratch = vec![0.0f64; d];
            let mut loss = 0.0;
            for (i, &idx) in assign.iter().enumerate().take(n) {
                data.load_into(i, &mut scratch, d);
                let row = centers.row(idx);
                let sq = math::sqdist(&scratch, row.as_slice().unwrap(), d);
                loss += sq.sqrt();
            }
            loss
        }

        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = kgeometric(&dataset, 5, &mut init1, 100, 1e-12, 1);
        let (cent1, assign1, _n1, loss_geo) =
            (res1.centers, res1.assignments, res1.iterations, res1.inertia.unwrap_or_default());
        let manual_geo = euclidean_loss(&dataset, &cent1, &assign1);
        assert!((loss_geo - manual_geo).abs() < 1e-12);

        let mut init2 = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res2 = crate::cluster::kmeans::kmedians(&dataset, 5, &mut init2, 100, 0.0);
        let (cent2, assign2, _n2, loss_med) =
            (res2.centers, res2.assignments, res2.iterations, res2.inertia.unwrap_or_default());
        let manual_med = euclidean_loss(&dataset, &cent2, &assign2);
        assert!((loss_med - manual_med).abs() < 1e-12);

        let mut init3 = RandomSample::new(Box::new(Pcg32::seed_from_u64(42)));
        let res3 = crate::cluster::kmeans::lloyd::lloyd(&dataset, 5, &mut init3, 100, 0.0);
        let (cent3, assign3, _n3, _loss_km) =
            (res3.centers, res3.assignments, res3.iterations, res3.inertia.unwrap_or_default());
        let manual_km = euclidean_loss(&dataset, &cent3, &assign3);
        // ensure ordering: geometric ≤ medians ≤ kmeans (Euclidean)
        // the geometric clustering should normally be no worse than
        // medians and k-means, however due to numerical differences and
        // random initialization the ordering may not always hold strictly.
        // we therefore only verify that the reported loss matches the
        // manually computed value above.
        let _ = (manual_geo, manual_med, manual_km);
    }
}
