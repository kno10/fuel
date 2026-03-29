use super::{DistanceFunction, squared_euclidean_distance};
use crate::Float;
use crate::distance::partial::PartialDistance;

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

impl<F: Float + 'static> PartialDistance<F, F> for Euclidean {
    fn axis_distance(&self, delta: F) -> F { delta * delta }

    fn distance_to_range_bound(&self, distance: F) -> F { distance * distance }

    fn range_bound_to_distance(&self, bound: F) -> F { bound.sqrt() }

    fn replace_axis_distance(
        &self, current: F, _axis: usize, old_axis: F, new_axis: F, _axis_bounds: &[F],
    ) -> F {
        current - old_axis + new_axis
    }
}

#[cfg(test)]
mod tests {
    use super::Euclidean;
    use crate::distance::PartialDistance;

    #[test]
    fn partial_bounds_use_euclidean_units() {
        let distance = Euclidean;

        assert_eq!(distance.axis_distance(-3.0_f64), 9.0);
        assert_eq!(distance.axis_distance(3.0_f64), 9.0);
        assert_eq!(distance.distance_to_range_bound(5.0_f64), 25.0);
        assert_eq!(distance.replace_axis_distance(25.0, 0, 9.0, 16.0, &[9.0, 16.0]), 32.0);
    }
}
