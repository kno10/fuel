use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::init::*;
use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// Standard k-means algorithm (Lloyd, Forgy) - naive textbook implementation
// Inline always to allow CPU optimization!
// Otherwise, CPU properties such as fma/avx2 may get lost and this will severely harm performance.
#[inline(always)]
fn lloyd_naive_impl<N, I, A>(
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
    init.init::<A>(data, &mut cent, k);
    let mut assign = vec![k; n];
    let mut iter = 0;
    let mut lastsum = N::infinity();
    while iter < maxiter {
        iter += 1;
        // compute old centers copy for tolerance if needed
        let old_cent = if tol > N::zero() { Some(cent.clone()) } else { None };
        let (mut changed, mut sum) = (0, N::zero());
        // Reassign each point to the nearest center
        for (i, assign_i) in assign.iter_mut().enumerate().take(n) {
            let aa = *assign_i;
            data.load_into(i, &mut scratch, d);
            let (mut a, mut s) = (k, N::infinity());
            for j in 0..k {
                let tmp = math::sqdist(cent.center(j), &scratch, d);
                if tmp < s {
                    (a, s) = (j, tmp);
                }
            }
            if a != aa {
                *assign_i = a;
                changed += 1;
            }
            sum += s;
        }
        lastsum = sum;
        if changed == 0 {
            break;
        }
        // Recompute centers
        let mut csize = vec![0_usize; k];
        for j in 0..k {
            cent.center_mut(j).fill(N::zero());
        }
        for i in 0..n {
            data.load_into(i, &mut scratch, d);
            math::add_assign(cent.center_mut(assign[i]), &scratch, d);
            csize[assign[i]] += 1;
        }
        for (j, &csize_j) in csize.iter().enumerate().take(k) {
            if csize_j == 0 {
                println!("Cluster has become empty, not handled in naive implementation!");
                continue;
            }
            math::mul_assign(cent.center_mut(j), N::from(csize_j).unwrap().recip(), d);
        }
        // tolerance check
        if let Some(ref old) = old_cent {
            let diff = cent.diff_frobenius_norm(old);
            let norm = old.frobenius_norm();
            let rel = if norm == N::zero() { diff } else { diff / norm };
            if rel <= tol {
                break;
            }
        }
    }
    KMeansResult::with_inertia(cent.into_ndarray(), assign, iter, lastsum)
}

pub fn lloyd_naive<N, I, A>(
    data: &A, k: usize, init: &mut I, maxiter: usize, tol: N,
) -> KMeansResult<N>
where
    N: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display + 'static,
    I: Initialization<N>,
    A: Dataset<N>,
{
    lloyd_naive_impl::<N, I, A>(data, k, init, maxiter, tol)
}

#[cfg(test)]
mod tests {
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
        let res = lloyd_naive_impl::<_, _, _>(&dataset, 5, &mut init, 100, 0.0);
        let loss = compute_loss(&dataset, &res.centers, &res.assignments);
        assert!(
            res.inertia.is_some() && (loss - res.inertia.unwrap()).abs() < 1e-12,
            "loss not correct"
        );
        assert!((loss - 50.82715291533402).abs() < 1e-12, "loss not as expected: {}", loss);
        assert_eq!(res.iterations, 11, "niter not as expected");
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_tolerance() {
        let mat = gen_test_data((100, 2), Pcg32::seed_from_u64(42));
        let dataset = NdArrayDataset::new(&mat);
        let mut init1 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res1 = lloyd_naive_impl::<_, _, _>(&dataset, 5, &mut init1, 100, 0.0);
        let n1 = res1.iterations;
        let tol: f64 = 1e-3;
        let mut init2 = RandomSample::new(Pcg32::seed_from_u64(42));
        let res2 = lloyd_naive_impl::<_, _, _>(&dataset, 5, &mut init2, 100, tol);
        let n2 = res2.iterations;
        assert!(n2 <= n1, "tolerance should not increase iteration count");
    }
}
