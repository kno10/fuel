/// Flexible-beta linkage with parameter β.
use num_traits::Float;

#[derive(Clone, Copy, Debug)]
pub struct FlexibleBetaLinkage<F: Float> {
    beta: F,
    alpha: F,
}

impl<F: Float> FlexibleBetaLinkage<F> {
    /// Create a new instance for given β.  α = (1-β)/2 as per definition.
    pub fn new(beta: F) -> Self {
        Self {
            beta,
            alpha: (F::one() - beta) / (F::one() + F::one()),
        }
    }
}

use super::Linkage;

impl<F: Float> Linkage<F> for FlexibleBetaLinkage<F> {
    fn combine(&self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, dxy: F) -> F {
        self.alpha * dx + self.alpha * dy + self.beta * dxy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn flexible_beta_defaults() {
        let fb: FlexibleBetaLinkage<f64> = FlexibleBetaLinkage::new(-0.25);
        assert_eq!(
            fb.combine(1, 1.0, 1, 1.0, 0, 2.0),
            fb.alpha * 1.0 + fb.alpha * 1.0 + fb.beta * 2.0
        );
    }

    #[test]
    fn flexible_beta_f32_compile() {
        let fb: FlexibleBetaLinkage<f32> = FlexibleBetaLinkage::new(-0.25);
        let r: f32 = fb.combine(1, 1.0_f32, 1, 1.0_f32, 0, 2.0_f32);
        assert!((r - (fb.alpha * 1.0 + fb.alpha * 1.0 + fb.beta * 2.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn agnes_with_flexible_beta_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, FlexibleBetaLinkage::new(-0.25), false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }
}
