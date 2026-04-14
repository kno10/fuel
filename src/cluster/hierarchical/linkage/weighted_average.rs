/// Weighted average linkage (WPGMA).
#[derive(Clone, Copy, Default, Debug)]
pub struct WeightedAverageLinkage;

use crate::Float;
use crate::cluster::hierarchical::Linkage;

impl<F: Float> Linkage<F> for WeightedAverageLinkage {
    fn combine(
        &self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, _dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        F::half() * (dx + dy)
    }
}
