use crate::Float;
use crate::distance::dot;
use crate::kernel::Kernel;

/// Sigmoid kernel: `tanh(c * <x, y> + theta)`.
///
/// $$k(x, y) = \tanh(c \langle x, y\rangle + \theta)$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmoidKernel<F: Float> {
    pub c: F,
    pub theta: F,
}

impl<F: Float> SigmoidKernel<F> {
    pub fn new(c: F, theta: F) -> Self { Self { c, theta } }

    pub fn similarity(&self, x: &[F], y: &[F]) -> F {
        (self.c * dot::<F, F>(x, y) + self.theta).tanh()
    }
}

impl<F: Float> Kernel<Vec<F>, F> for SigmoidKernel<F> {
    fn similarity(&self, x: &Vec<F>, y: &Vec<F>) -> F {
        self.similarity(x.as_slice(), y.as_slice())
    }
}

impl<F: Float> Kernel<[F], F> for SigmoidKernel<F> {
    fn similarity(&self, x: &[F], y: &[F]) -> F { self.similarity(x, y) }
}
