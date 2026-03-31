use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign, SubAssign};

use rand::Rng;

use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset};

/// Random sampling initialization, very basic
pub struct RandomSample<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
{
    rng: R,
    phantom: PhantomData<N>,
}

impl<N, R> RandomSample<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
    R: Rng,
{
    pub fn new(rng: R) -> Self { Self { rng, phantom: PhantomData } }
}

impl<N, R> super::Initialization<N> for RandomSample<N, R>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
    R: Rng,
{
    fn uses_distances(&self) -> bool { false }

    fn init<A>(&mut self, data: &A, cent: &mut Centers<N>, k: usize)
    where
        A: Dataset<N>,
    {
        let (n, d) = (data.nrows(), data.ncols());
        for (i, x) in rand::seq::index::sample(&mut self.rng, n, k).iter().enumerate() {
            data.load_into(x, cent.center_mut(i), d);
        }
    }

    #[allow(unused_variables)]
    fn init_with_distances<A, F>(
        &mut self, data: &A, cent: &mut Centers<N>, k: usize, callback: Option<F>,
    ) where
        A: Dataset<N>,
        F: FnMut(usize, usize, N),
    {
        panic!("Distances not provided by this initialization.");
    }
}
