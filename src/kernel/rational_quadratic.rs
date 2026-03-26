use crate::Float;
use crate::distance::squared_euclidean_distance;
use crate::kernel::Kernel;

/// Rational Quadratic kernel: 1 - d^2 / (d^2 + c), with d^2 = ||x - y||^2.
///
/// $$k(x, y) = 1 - \frac{\|x - y\|^2}{\|x - y\|^2 + c}$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RationalQuadraticKernel<F: Float> {
    pub c: F,
}

impl<F: Float> RationalQuadraticKernel<F> {
    pub fn new(c: F) -> Self { Self { c } }

    pub fn similarity(&self, x: &[F], y: &[F]) -> F {
        let dist2 = squared_euclidean_distance::<F, F>(x, y);
        F::one() - dist2 / (dist2 + self.c)
    }
}

impl<F: Float> Kernel<Vec<F>, F> for RationalQuadraticKernel<F> {
    fn similarity(&self, x: &Vec<F>, y: &Vec<F>) -> F {
        self.similarity(x.as_slice(), y.as_slice())
    }
}

impl<F: Float> Kernel<[F], F> for RationalQuadraticKernel<F> {
    fn similarity(&self, x: &[F], y: &[F]) -> F { self.similarity(x, y) }
}
