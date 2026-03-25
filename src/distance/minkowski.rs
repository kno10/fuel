use crate::Float;
use crate::distance::DistanceFunction;
use crate::distance::partial::PartialDistance;

/// Minkowski distance with exponent `p > 0`.
#[derive(Debug, Clone, Copy)]
pub struct MinkowskiDistance<F> {
    p: F,
}

impl<F: Float> MinkowskiDistance<F> {
    /// Create a new Minkowski distance for exponent `p`.
    pub fn new(p: F) -> Self {
        assert!(p > F::zero(), "Minkowski exponent must be positive");
        Self { p }
    }
}

impl<F: Float> MinkowskiDistance<F> {
    fn distance_impl(&self, a: &[F], b: &[F]) -> F {
        let mut accum = F::zero();
        for (&x, &y) in a.iter().zip(b.iter()) {
            accum = accum + (x - y).abs().powf(self.p);
        }
        accum.powf(F::one() / self.p)
    }
}

impl<F: Float> DistanceFunction<[F], F> for MinkowskiDistance<F> {
    fn distance(&self, a: &[F], b: &[F]) -> F { self.distance_impl(a, b) }
}

impl<F: Float> DistanceFunction<Vec<F>, F> for MinkowskiDistance<F> {
    fn distance(&self, a: &Vec<F>, b: &Vec<F>) -> F {
        self.distance_impl(a.as_slice(), b.as_slice())
    }
}

impl<F: Float + Copy> PartialDistance<F, F> for MinkowskiDistance<F> {
    fn axis_distance(&self, delta: F) -> F { delta.abs().powf(self.p) }

    fn combine_axis_distances(&self, a: F, b: F) -> F { a.max(b) }
}
