use super::{DistanceFunction, squared_euclidean_distance};
use crate::Float;
use crate::distance::partial::Partial;

/// Euclidean distance (L2 norm):
/// $$d_2(a,b)=\sqrt{\sum_i (a_i-b_i)^2}$$
pub fn euclidean_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    squared_euclidean_distance::<N, F>(a, b).sqrt()
}

#[derive(Debug, Clone, Copy, Default)]
/// Euclidean distance strategy (standard L2).
pub struct Euclidean;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for Euclidean {
    fn distance(&self, a: &[N], b: &[N]) -> F { euclidean_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}

impl<N: Float, F: Float + 'static> DistanceFunction<Vec<N>, F> for Euclidean {
    fn distance(&self, a: &Vec<N>, b: &Vec<N>) -> F { euclidean_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}

impl<F: Float + 'static> Partial<F, F> for Euclidean {
    fn axis_distance(&self, delta: F) -> F { delta * delta }

    fn combine_axis_distances(&self, a: F, b: F) -> F { a + b }
}
