/// Sigmoid kernel: tanh(c * <x,y> + theta)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SigmoidKernel {
    pub c: f64,
    pub theta: f64,
}

impl SigmoidKernel {
    pub fn new(c: f64, theta: f64) -> Self { Self { c, theta } }

    pub fn similarity(&self, x: &[f64], y: &[f64]) -> f64 {
        let dot: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        (self.c * dot + self.theta).tanh()
    }
}
