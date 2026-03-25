use crate::Float;
use crate::distance::{DistanceFunction, DistanceMetric};

pub fn chebyshev_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let left: F = (*x).to_float::<F>();
            let right: F = (*y).to_float::<F>();
            (left - right).abs()
        })
        .fold(F::zero(), F::max)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ChebyshevDistance;

impl<N: Float, F: Float + 'static> DistanceMetric<[N], F> for ChebyshevDistance {}

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for ChebyshevDistance {
    fn distance(&self, a: &[N], b: &[N]) -> F { chebyshev_distance(a, b) }
}
