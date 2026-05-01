use std::iter::Sum;
use std::ops::*;

use ndarray::Array2;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::lloyd::lloyd_initial_assignment;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Internal generic implementation of fuzzy k-means using Lloyd-style iterations.
#[inline(always)]
pub fn fuzzy_lloyd<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, m: N,
) -> (Array2<N>, Array2<N>, Vec<usize>, usize, N)
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    assert!(m > N::one(), "fuzziness exponent must be > 1");
    let (n, d) = (data.nrows(), data.ncols());
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);

    // start with a crisp initial assignment to seed both centers and U
    let (mut assign, _csize, _lastsum) = lloyd_initial_assignment::<N, A, I>(
        data,
        None,
        k,
        init,
        &mut cent,
        &mut sums,
        &mut scratch,
    );

    // initialize membership matrix U (row-major n*k)
    let mut u = vec![N::zero(); n * k];
    for i in 0..n {
        u[i * k + assign[i]] = N::one();
    }

    let one = N::one();
    let m1 = m - one;
    let expo = one / m1;

    let mut iter = 0;
    let mut loss = N::zero();

    while iter < maxiter {
        iter += 1;

        // update centers using weighted average of points with weights u_{ij}^m
        for j in 0..k {
            for v in sums.center_mut(j).iter_mut() {
                *v = N::zero();
            }
            let mut denom = N::zero();
            for i in 0..n {
                let weight = u[i * k + j].powf(m);
                if weight != N::zero() {
                    data.load_into(i, &mut scratch, d);
                    math::axpy(sums.center_mut(j), weight, &scratch, d);
                    denom += weight;
                }
            }
            if denom != N::zero() {
                math::mul(cent.center_mut(j), sums.center(j), denom.recip(), d);
            }
        }

        // update memberships and compute loss; also track crisp assignment changes
        let mut changed = 0;
        loss = N::zero();
        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            let mut dists = vec![N::zero(); k];
            let mut zero_idx: Option<usize> = None;
            for (j, dslot) in dists.iter_mut().enumerate().take(k) {
                let dist = math::sqdist(cent.center(j), &scratch, d);
                *dslot = dist;
                if dist.is_zero() {
                    zero_idx = Some(j);
                }
            }
            if let Some(t) = zero_idx {
                for j in 0..k {
                    u[i * k + j] = if j == t { one } else { N::zero() };
                }
                if assign[i] != t {
                    assign[i] = t;
                    changed += 1;
                }
            } else {
                let mut best_u = N::zero();
                let mut best_j = 0;
                for j in 0..k {
                    let mut sum = N::zero();
                    for l in 0..k {
                        let ratio = dists[j] / dists[l];
                        sum += ratio.powf(expo);
                    }
                    let new_u = one / sum;
                    u[i * k + j] = new_u;
                    if new_u > best_u {
                        best_u = new_u;
                        best_j = j;
                    }
                    loss += new_u.powf(m) * dists[j];
                }
                if best_j != assign[i] {
                    assign[i] = best_j;
                    changed += 1;
                }
            }
        }
        if changed == 0 {
            break;
        }
    }

    let centers_arr = cent.into_ndarray();
    let membership_arr = Array2::from_shape_vec((n, k), u).unwrap();
    (centers_arr, membership_arr, assign, iter, loss)
}

/// Public fuzzy k-means wrapper with runtime dispatch
#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::util::gen_test_data;

    #[test]
    fn test_fuzzy() {
        let mat = gen_test_data((100, 2), Box::new(Pcg32::seed_from_u64(123)));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Box::new(Pcg32::seed_from_u64(123)));
        let m = 2.0_f64;
        let (cent, members, assign, niter, los) = fuzzy_lloyd(&dataset, 5, &mut init, 100, m);
        let loss2 = crate::cluster::kmeans::util::compute_fuzzy_loss::<_, _, _>(
            &dataset, &cent, &members, m,
        );
        assert!((loss2 - los).abs() < 1e-12, "fuzzy loss not correct");
        let mut assign2 = Vec::with_capacity(members.nrows());
        for row in members.rows() {
            let (best_j, _) =
                row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
            assign2.push(best_j);
        }
        assert_eq!(assign, assign2);
        assert!(niter > 0, "should perform at least one iteration");
    }
}
