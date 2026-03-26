use crate::Float;
use crate::distance::DistanceFunction;
use crate::distance::partial::Partial;

/// Minkowski distance (L^p norm):
/// $$d_p(a,b)=\left(\sum_i |a_i-b_i|^p\right)^{1/p}$$
pub fn minkowski_distance<F: Float>(a: &[F], b: &[F], p: F) -> F {
    assert!(p > F::zero(), "Minkowski exponent must be positive");

    let mut accum = F::zero();
    for (&x, &y) in a.iter().zip(b.iter()) {
        accum = accum + (x - y).abs().powf(p);
    }
    accum.powf(F::one() / p)
}

/// Minkowski distance with exponent `p > 0`.
#[derive(Debug, Clone, Copy)]
/// Minkowski distance strategy (general Lp norm).
pub struct Minkowski<F> {
    p: F,
}

impl<F: Float> Minkowski<F> {
    /// Create a new Minkowski distance for exponent `p`.
    pub fn new(p: F) -> Self {
        assert!(p > F::zero(), "Minkowski exponent must be positive");
        Self { p }
    }
}

impl<F: Float> DistanceFunction<[F], F> for Minkowski<F> {
    fn distance(&self, a: &[F], b: &[F]) -> F { minkowski_distance(a, b, self.p) }

    fn is_metric(&self) -> bool { self.p >= F::one() }
}

impl<F: Float> DistanceFunction<Vec<F>, F> for Minkowski<F> {
    fn distance(&self, a: &Vec<F>, b: &Vec<F>) -> F {
        minkowski_distance(a.as_slice(), b.as_slice(), self.p)
    }

    fn is_metric(&self) -> bool { self.p >= F::one() }
}

impl<F: Float + Copy> Partial<F, F> for Minkowski<F> {
    fn axis_distance(&self, delta: F) -> F { delta.abs().powf(self.p) }

    fn combine_axis_distances(&self, a: F, b: F) -> F { a.max(b) }
}
