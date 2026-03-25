/// Radial Basis Function kernel (Gaussian RBF): exp(-gamma * ||x - y||^2)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialBasisFunctionKernel {
    pub gamma: f64,
}

impl RadialBasisFunctionKernel {
    pub fn new(gamma: f64) -> Self { Self { gamma } }

    pub fn similarity(&self, x: &[f64], y: &[f64]) -> f64 {
        let dist2: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| {
                let v = a - b;
                v * v
            })
            .sum();
        (-self.gamma * dist2).exp()
    }
}
