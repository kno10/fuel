/// Flexible-beta linkage with parameter β.
///
/// The recurrence is defined by
/// $\alpha dx + \alpha dy + \beta d_{xy}$ with
/// $\alpha = \frac{1-\beta}{2}$.
/// This method produces inversions when $\beta < 0$ and is supported by
/// stored-matrix algorithms.
use crate::Float;

#[derive(Clone, Copy, Debug)]
pub struct FlexibleBetaLinkage<F: Float> {
    beta: F,
    alpha: F,
}

impl<F: Float> FlexibleBetaLinkage<F> {
    /// Create a new instance for given β.  α = (1-β)/2 as per definition.
    pub fn new(beta: F) -> Self { Self { beta, alpha: (F::one() - beta) / F::two() } }
}

use crate::cluster::hierarchical::Linkage;

impl<F: Float> Linkage<F> for FlexibleBetaLinkage<F> {
    fn can_produce_inversions(&self) -> bool { self.beta < F::zero() }

    fn combine(
        &self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        self.alpha * dx + self.alpha * dy + self.beta * dxy
    }
}
