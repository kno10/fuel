/// Linear kernel (dot product) with optional coef0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearKernel {
    pub coef0: f64,
}

impl LinearKernel {
    pub fn new(coef0: f64) -> Self { Self { coef0 } }

    pub fn similarity(&self, x: &[f64], y: &[f64]) -> f64 {
        let dot: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        dot + self.coef0
    }
}
