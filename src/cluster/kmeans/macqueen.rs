use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Historic MacQueen k-means (sequential update)
#[inline(always)]
pub fn macqueen<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    I: Initialization<N>,
    A: Dataset<N>,
{
    let d = data.ncols();
    let mut scratch = vec![N::zero(); d];
    let mut cent = Centers::<N>::new(k, d);
    let mut sums = Centers::<N>::new(k, d);

    let (mut assign, mut csize, mut lastsum) =
        crate::cluster::kmeans::lloyd::lloyd_initial_assignment::<N, A, I>(
            data,
            None,
            k,
            init,
            &mut cent,
            &mut sums,
            &mut scratch,
        );

    let mut iter = 1; // initial assignment counts as first iteration

    while iter < maxiter {
        iter += 1;

        let old_cent = if tol > N::zero() { Some(cent.clone()) } else { None };

        let mut changed = 0;

        // Process points sequentially, updating centers immediately.
        let assign_clone = assign.clone();
        for (i, aa) in assign_clone.into_iter().enumerate() {
            data.load_into(i, &mut scratch, d);
            let (mut a, mut s) = (0, math::sqdist(cent.center(0), &scratch, d));
            for j in 1..k {
                let tmp = math::sqdist(cent.center(j), &scratch, d);
                if tmp < s {
                    (a, s) = (j, tmp);
                }
            }
            if a != aa {
                // remove from old cluster
                if csize[aa] > 1 {
                    csize[aa] -= 1;
                    math::sub_assign(sums.center_mut(aa), &scratch, d);
                    let recip = N::from(csize[aa]).unwrap().recip();
                    math::mul(&mut scratch, sums.center(aa), recip, d);
                    math::copy(cent.center_mut(aa), &scratch, d);
                } else {
                    // cluster becomes empty
                    csize[aa] = 0;
                    for v in cent.center_mut(aa).iter_mut() {
                        *v = N::zero();
                    }
                    for v in sums.center_mut(aa).iter_mut() {
                        *v = N::zero();
                    }
                }

                // add to new cluster
                csize[a] += 1;
                math::add_assign(sums.center_mut(a), &scratch, d);
                let recip = N::from(csize[a]).unwrap().recip();
                math::mul(&mut scratch, sums.center(a), recip, d);
                math::copy(cent.center_mut(a), &scratch, d);

                assign[i] = a;
                changed += 1;
            }
        }

        // compute current inertia (sum of squared distances to current centers)
        let mut sum = N::zero();
        for (i, &assign_i) in assign.iter().enumerate() {
            data.load_into(i, &mut scratch, d);
            sum += math::sqdist(cent.center(assign_i), &scratch, d);
        }
        lastsum = sum;

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

    KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum)
}

/// Classic MacQueen k-means.
#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand_pcg::Pcg32;

    use super::*;
    use crate::NdArrayDataset;
    use crate::cluster::kmeans::util::{compute_loss, gen_test_data};

    #[test]
    fn test_basic() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init = RandomSample::new(Pcg32::seed_from_u64(42));
        let res = macqueen(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert!(res.inertia.is_some() && (loss - res.inertia.unwrap()).abs() < 1e-12);
        assert!(res.iterations <= 100);
    }
}
