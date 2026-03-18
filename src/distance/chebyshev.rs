use num_traits::{AsPrimitive, Float, ToPrimitive};

use super::{DistanceFunction, DistanceMetric};

pub fn chebyshev_distance<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let left: F = (*x).as_();
            let right: F = (*y).as_();
            (left - right).abs()
        })
        .fold(F::zero(), F::max)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChebyshevDistance;

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceMetric<[N], F>
    for ChebyshevDistance
{
}

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceFunction<[N], F>
    for ChebyshevDistance
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        chebyshev_distance(a, b)
    }
}
