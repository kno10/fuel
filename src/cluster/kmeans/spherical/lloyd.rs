use ndarray::Array2;

use crate::cluster::kmeans::Centers;
use crate::cluster::kmeans::init::*;
use crate::{Float, VectorData as Dataset, math};

/// Standard spherical k-means algorithm (Lloyd, Forgy with cosine similarity)
#[inline(always)]
pub fn spherical_lloyd<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> (Array2<N>, Vec<usize>, usize, N)
where
    N: Float,
    I: Initialization<N>,
    A: Dataset<N>,
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
    let mut scratch = vec![N::zero(); d];
    for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
        data.load_into(i, &mut scratch, d);
        let (mut a, mut s) = (0, math::dot(&scratch, cent.center(0), d));
        for j in 1..k {
            let tmp = math::dot(&scratch, cent.center(j), d);
            if tmp > s {
                (a, s) = (j, tmp);
            }
        }
        csize[a] += 1;
        *assign_i = a;
        math::add_assign(sums.center_mut(a), &scratch, d);
        lastsum += s;
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
        let (mut changed, mut sum) = (0, N::zero());
        for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
            data.load_into(i, &mut scratch, d);
            let aa = *assign_i;
            let (mut a, mut s) = (0, math::dot(&scratch, cent.center(0), d));
            for j in 1..k {
                let tmp = math::dot(&scratch, cent.center(j), d);
                if tmp > s || (j == aa && tmp == s) {
                    (a, s) = (j, tmp);
                }
            }
            if a != aa {
                *assign_i = a;
                csize[aa] -= 1;
                csize[a] += 1;
                math::sub_assign(sums.center_mut(aa), &scratch, d);
                math::add_assign(sums.center_mut(a), &scratch, d);
                changed += 1;
            }
            sum += s;
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

    use crate::cluster::kmeans::init::FirstK;
    use crate::cluster::kmeans::ndarray::NdArrayDataset;
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
            let nrm = ((cent[[j, 0]] as f64) * (cent[[j, 0]] as f64)
                + (cent[[j, 1]] as f64) * (cent[[j, 1]] as f64))
                .sqrt();
            assert!((nrm - 1.0).abs() < 1e-12, "center is not normalized");
        }
    }
}
