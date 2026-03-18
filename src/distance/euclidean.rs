use num_traits::{AsPrimitive, Float, ToPrimitive};

use super::{DistanceFunction, DistanceMetric, squared_euclidean_distance};
use crate::distance::partial::PartialDistance;

pub fn euclidean_distance<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    squared_euclidean_distance::<N, F>(a, b).sqrt()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanDistance;

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceMetric<[N], F>
    for EuclideanDistance
{
}

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceFunction<[N], F>
    for EuclideanDistance
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        euclidean_distance(a, b)
    }
}

// FIXME: redundant?
impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceMetric<Vec<N>, F>
    for EuclideanDistance
{
}

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceFunction<Vec<N>, F>
    for EuclideanDistance
{
    fn distance(&self, a: &Vec<N>, b: &Vec<N>) -> F {
        euclidean_distance(a, b)
    }
}

impl<F: Float + Copy> PartialDistance<F> for EuclideanDistance {
    fn distance(&self, a: &[F], b: &[F]) -> F {
        a.iter()
            .zip(b)
            .fold(F::zero(), |acc, (&x, &y)| {
                let delta = x - y;
                acc + delta * delta
            })
            .sqrt()
    }
}
