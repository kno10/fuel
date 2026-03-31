use std::marker::PhantomData;
use std::ops::{AddAssign, MulAssign, SubAssign};

use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset};

/// First K initialization
#[derive(Clone)]
pub struct FirstK<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
{
    phantom: PhantomData<N>,
}

impl<N> FirstK<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
{
    pub fn new() -> Self { Self { phantom: PhantomData } }
}

impl<N> Default for FirstK<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
{
    fn default() -> Self { Self::new() }
}

impl<N> super::Initialization<N> for FirstK<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
{
    fn uses_distances(&self) -> bool { false }

    fn init<A>(&mut self, data: &A, cent: &mut Centers<N>, k: usize)
    where
        A: Dataset<N>,
    {
        let d = data.ncols();
        for i in 0..k {
            data.load_into(i, cent.center_mut(i), d);
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
