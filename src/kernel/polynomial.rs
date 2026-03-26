use crate::Float;
use crate::distance::dot;
use crate::kernel::Kernel;

/// Polynomial kernel in the form `(gamma * <x, y> + coef0)^degree`.
///
/// $$k(x, y) = (\gamma \langle x, y\rangle + c)^d$$
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialKernel<F: Float> {
    pub degree: usize,
    pub gamma: F,
    pub coef0: F,
}

impl<F: Float> PolynomialKernel<F> {
    pub fn new(degree: usize, gamma: F, coef0: F) -> Self { Self { degree, gamma, coef0 } }

    pub fn similarity(&self, x: &[F], y: &[F]) -> F {
        let dot = dot::<F, F>(x, y);
        (self.gamma * dot + self.coef0).powi(self.degree as i32)
    }
}

impl<F: Float> Kernel<Vec<F>, F> for PolynomialKernel<F> {
    fn similarity(&self, x: &Vec<F>, y: &Vec<F>) -> F {
        self.similarity(x.as_slice(), y.as_slice())
    }
}

impl<F: Float> Kernel<[F], F> for PolynomialKernel<F> {
    fn similarity(&self, x: &[F], y: &[F]) -> F { self.similarity(x, y) }
}
