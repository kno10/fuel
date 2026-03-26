use crate::Float;
use crate::distance::squared_euclidean_distance;
use crate::kernel::Kernel;

/// Radial Basis Function kernel (Gaussian RBF): exp(-gamma * ||x - y||^2).
///
/// $$k(x, y) = \exp(-\gamma \|x - y\|^2) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialBasisFunctionKernel<F: Float> {
    pub gamma: F,
}

impl<F: Float> RadialBasisFunctionKernel<F> {
    pub fn new_gamma(gamma: F) -> Self { Self { gamma } }

    pub fn new_sigma(sigma: F) -> Self { Self { gamma: F::from(0.5).unwrap() / (sigma * sigma) } }

    pub fn similarity(&self, x: &[F], y: &[F]) -> F {
        (-self.gamma * squared_euclidean_distance::<F, F>(x, y)).exp()
    }
}

impl<F: Float> Kernel<Vec<F>, F> for RadialBasisFunctionKernel<F> {
    fn similarity(&self, x: &Vec<F>, y: &Vec<F>) -> F {
        self.similarity(x.as_slice(), y.as_slice())
    }
}

impl<F: Float> Kernel<[F], F> for RadialBasisFunctionKernel<F> {
    fn similarity(&self, x: &[F], y: &[F]) -> F { self.similarity(x, y) }
}
