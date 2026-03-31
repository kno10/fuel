use crate::Float;
use crate::distance::DistanceFunction;
use crate::distance::partial::PartialDistance;

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

impl<F: Float> PartialDistance<F, F> for Minkowski<F> {
    fn axis_distance(&self, delta: F) -> F { delta.abs().powf(self.p) }

    fn distance_to_range_bound(&self, distance: F) -> F { distance }

    fn range_bound_to_distance(&self, bound: F) -> F { bound }

    fn replace_axis_distance(
        &self, current: F, _axis: usize, old_axis: F, new_axis: F, axis_bounds: &[F],
    ) -> F {
        // For max-based partial bound, if the current maximum is replaced,
        // recompute by scanning axis_bounds.
        if old_axis < new_axis {
            if new_axis >= current {
                return new_axis;
            }
            return current;
        }

        if old_axis == current {
            axis_bounds.iter().copied().fold(F::zero(), |acc, x| acc.max(x))
        } else {
            current
        }
    }
}
