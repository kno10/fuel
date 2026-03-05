/// Weighted average linkage (WPGMA).
#[derive(Clone, Copy, Default, Debug)]
pub struct WeightedAverageLinkage;

use super::Linkage;
use num_traits::Float;

impl<F: Float> Linkage<F> for WeightedAverageLinkage {
    fn combine(&self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, _dxy: F) -> F {
        F::from(0.5).unwrap() * (dx + dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn weighted_average_behaves_as_half() {
        let w = WeightedAverageLinkage;
        assert_eq!(w.combine(1, 1.0, 1, 3.0, 0, 0.0), 2.0);
    }

    #[test]
    fn agnes_with_weighted_average_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, WeightedAverageLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn weighted_average_f32_compile() {
        let w = WeightedAverageLinkage;
        let r: f32 = w.combine(0, 1.0_f32, 0, 3.0_f32, 0, 0.0_f32);
        assert_eq!(r, 2.0_f32);
    }
}
