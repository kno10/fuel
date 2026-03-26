use crate::Float;
use crate::distance::DistanceFunction;

/// Chebyshev distance:
/// $$d_{\infty}(a,b)=\max_i |a_i-b_i|$$
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
/// Chebyshev max-norm distance strategy.
pub struct Chebyshev;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for Chebyshev {
    fn distance(&self, a: &[N], b: &[N]) -> F { chebyshev_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}
