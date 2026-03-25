use super::{DistanceFunction, DistanceMetric, squared_euclidean_distance};
use crate::Float;
use crate::distance::partial::PartialDistance;

pub fn euclidean_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let sq = squared_euclidean_distance::<N, F>(a, b);
    sq.sqrt()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EuclideanDistance;

impl<N: Float, F: Float + 'static> DistanceMetric<[N], F> for EuclideanDistance {}

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for EuclideanDistance {
    fn distance(&self, a: &[N], b: &[N]) -> F { euclidean_distance(a, b) }
}

impl<N: Float, F: Float + 'static> DistanceMetric<Vec<N>, F> for EuclideanDistance {}

impl<N: Float, F: Float + 'static> DistanceFunction<Vec<N>, F> for EuclideanDistance {
    fn distance(&self, a: &Vec<N>, b: &Vec<N>) -> F { euclidean_distance(a, b) }
}

impl<F: Float + 'static> PartialDistance<F, F> for EuclideanDistance {
    fn axis_distance(&self, delta: F) -> F { delta * delta }

    fn combine_axis_distances(&self, a: F, b: F) -> F { a + b }
}
