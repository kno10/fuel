use crate::Float;
use crate::distance::squared_euclidean_distance;
use crate::kernel::Kernel;

/// Laplace kernel: exp(-gamma * sqrt(||x - y||^2)), with gamma = 0.5 / sigma^2.
///
/// $$k(x, y) = \exp\left(-\frac{0.5}{\sigma^2} \|x - y\|\right)$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceKernel<F: Float> {
    pub gamma: F,
}

impl<F: Float> LaplaceKernel<F> {
    pub fn new_gamma(gamma: F) -> Self { Self { gamma } }

    pub fn new_sigma(sigma: F) -> Self {
        // gamma = 0.5 / sigma^2
        Self { gamma: F::from(0.5).unwrap() / (sigma * sigma) }
    }

    pub fn similarity(&self, x: &[F], y: &[F]) -> F {
        (-self.gamma * squared_euclidean_distance::<F, F>(x, y).sqrt()).exp()
    }
}

impl<F: Float> Kernel<Vec<F>, F> for LaplaceKernel<F> {
    fn similarity(&self, x: &Vec<F>, y: &Vec<F>) -> F {
        self.similarity(x.as_slice(), y.as_slice())
    }
}

impl<F: Float> Kernel<[F], F> for LaplaceKernel<F> {
    fn similarity(&self, x: &[F], y: &[F]) -> F { self.similarity(x, y) }
}
