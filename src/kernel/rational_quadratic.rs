/// Rational Quadratic kernel: 1 - d^2 / (d^2 + c), where d^2 = ||x - y||^2
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RationalQuadraticKernel {
    pub c: f64,
}

impl RationalQuadraticKernel {
    pub fn new(c: f64) -> Self { Self { c } }

    pub fn similarity(&self, x: &[f64], y: &[f64]) -> f64 {
        let dist2: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| {
                let v = a - b;
                v * v
            })
            .sum();
        1.0 - dist2 / (dist2 + self.c)
    }
}
