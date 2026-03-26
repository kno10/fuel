use crate::Float;
use crate::distance::dot;
use crate::kernel::Kernel;

/// Linear kernel (dot product) with optional `coef0` offset.
///
/// $$k(x, y) = \langle x, y\rangle + c$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearKernel<F: Float> {
    pub coef0: F,
}

impl<F: Float> LinearKernel<F> {
    pub fn new(coef0: F) -> Self { Self { coef0 } }

    pub fn similarity(&self, x: &[F], y: &[F]) -> F { dot::<F, F>(x, y) + self.coef0 }
}

impl<F: Float> Kernel<Vec<F>, F> for LinearKernel<F> {
    fn similarity(&self, x: &Vec<F>, y: &Vec<F>) -> F {
        self.similarity(x.as_slice(), y.as_slice())
    }
}

impl<F: Float> Kernel<[F], F> for LinearKernel<F> {
    fn similarity(&self, x: &[F], y: &[F]) -> F { self.similarity(x, y) }
}
