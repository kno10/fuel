use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign, SubAssign};

use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng, rng};

use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset, math};

/// k-geometric++ initialization (weights proportional to Euclidean, not sq.).
/// Used by the k-geometric median algorithm.
pub struct KGeometricPP<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
{
    rng: R,
    phantom: PhantomData<N>,
}

impl<N, R> KGeometricPP<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
    R: Rng,
{
    pub fn new(rng: R) -> Self { Self { rng, phantom: PhantomData } }
}

impl<N> Default for KGeometricPP<N, StdRng>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
{
    fn default() -> Self {
        let mut seed_rng = rng();
        let rng = StdRng::from_rng(&mut seed_rng);
        KGeometricPP::new(rng)
    }
}

impl<N, R> super::Initialization<N> for KGeometricPP<N, R>
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
        let mut scratch = vec![N::zero(); d];
        data.load_into(self.rng.random_range(0..n), cent.center_mut(0), d);
        let mut dsq = vec![N::infinity(); n];
        for i in 0..k - 1 {
            let mut sum = N::zero();
            let last = &cent.center(i);
            for (j, dsq_j) in dsq.iter_mut().enumerate().take(n) {
                data.load_into(j, &mut scratch, d);
                let dj = math::sqdist(last, &scratch, d);
                *dsq_j = N::min(*dsq_j, dj);
                // weight proportional to distance
                sum += dsq_j.sqrt();
                if let Some(cb) = callback.as_mut() {
                    cb(i, j, dj);
                }
            }
            let c;
            'outer: loop {
                let mut r = self.rng.random::<N>() * sum;
                for (j, &dsq_j) in dsq.iter().enumerate().take(n) {
                    r -= dsq_j.sqrt();
                    if r < N::zero() {
                        c = j;
                        break 'outer;
                    }
                }
                sum -= r;
            }
            data.load_into(c, cent.center_mut(i + 1), d);
        }
        let last = &cent.center(k - 1);
        for j in 0..n {
            data.load_into(j, &mut scratch, d);
            if let Some(cb) = callback.as_mut() {
                cb(k - 1, j, math::sqdist(last, &scratch, d));
            }
        }
    }
}
