/// Laplace kernel: exp(-gamma * sqrt(||x - y||^2)), gamma = 0.5/sigma^2
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LaplaceKernel {
    pub sigma: f64,
}

impl LaplaceKernel {
    pub fn new(sigma: f64) -> Self { Self { sigma } }

    pub fn similarity(&self, x: &[f64], y: &[f64]) -> f64 {
        let dist2: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| {
                let v = a - b;
                v * v
            })
            .sum();
        (-0.5 / (self.sigma * self.sigma) * dist2.sqrt()).exp()
    }
}
