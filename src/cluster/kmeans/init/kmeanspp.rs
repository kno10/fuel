use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign, SubAssign};

use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng, rng};

use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset};

/// k-means++ initialization
pub struct KMeansPP<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
{
    rng: R,
    phantom: PhantomData<N>,
}

impl<N, R> KMeansPP<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
    R: Rng,
{
    pub fn new(rng: R) -> Self { Self { rng, phantom: PhantomData } }
}

impl<N> Default for KMeansPP<N, StdRng>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
{
    /// Default initialization uses an OS-seeded `StdRng`.
    fn default() -> Self {
        let mut seed_rng = rng();
        let rng = StdRng::from_rng(&mut seed_rng);
        KMeansPP::new(rng)
    }
}

impl<N, R> super::Initialization<N> for KMeansPP<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
    R: Rng,
{
    fn uses_distances(&self) -> bool { true }

    #[inline(always)]
    fn init<A>(&mut self, data: &A, cent: &mut Centers<N>, k: usize)
    where
        A: Dataset<N>,
    {
        // default initializer does not request distance callbacks
        // annotate the `None` so the compiler can infer `F` correctly
        self.init_with_distances::<A, _>(data, cent, k, None::<fn(usize, usize, N)>)
    }

    #[inline(always)]
    fn init_with_distances<A, F>(
        &mut self, data: &A, cent: &mut Centers<N>, k: usize, mut callback: Option<F>,
    ) where
        A: Dataset<N>,
        F: FnMut(usize, usize, N),
    {
        let (n, d) = (data.nrows(), data.ncols());
        data.load_into(self.rng.random_range(0..n), cent.center_mut(0), d);
        let mut dsq = vec![N::infinity(); n];

        let rows = data.to_ndarray();
        let mut tmp_dists = vec![N::zero(); n];

        for i in 0..k - 1 {
            N::vec_row_sqdist(cent.center(i), rows.view(), d, &mut tmp_dists, n);
            let mut sum = N::zero();
            for (j, dsq_j) in dsq.iter_mut().enumerate() {
                let dj = tmp_dists[j];
                if let Some(cb) = callback.as_mut() {
                    cb(i, j, dj);
                }
                *dsq_j = N::min(*dsq_j, dj);
                sum += *dsq_j;
            }
            let c;
            'outer: loop {
                let mut r = self.rng.random::<N>() * sum;
                for (j, &dsq_j) in dsq.iter().enumerate().take(n) {
                    r -= dsq_j;
                    if r < N::zero() {
                        c = j;
                        break 'outer;
                    }
                }
                sum -= r;
            }
            cent.center_mut(i + 1)[..d].copy_from_slice(rows.row(c).to_slice().unwrap());
        }
        if callback.is_some() {
            N::vec_row_sqdist(cent.center(k - 1), rows.view(), d, &mut tmp_dists, n);
            for j in 0..n {
                if let Some(cb) = callback.as_mut() {
                    cb(k - 1, j, tmp_dists[j]);
                }
            }
        }
    }
}
