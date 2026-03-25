/// Polynomial kernel in the form (gamma * <x, y> + coef0)^degree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialKernel {
    pub degree: usize,
    pub gamma: f64,
    pub coef0: f64,
}

impl PolynomialKernel {
    pub fn new(degree: usize, gamma: f64, coef0: f64) -> Self { Self { degree, gamma, coef0 } }

    pub fn similarity(&self, x: &[f64], y: &[f64]) -> f64 {
        let dot: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        (self.gamma * dot + self.coef0).powi(self.degree as i32)
    }
}
