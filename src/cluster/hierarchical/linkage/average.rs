/// Average-linkage criterion (weighted mean of distances).
#[derive(Clone, Copy, Default, Debug)]
pub struct AverageLinkage;

use num_traits::Float;
use super::Linkage;

impl<F: Float> Linkage<F> for AverageLinkage {
    fn combine(
        &self,
        sizex: usize,
        dx: F,
        sizey: usize,
        dy: F,
        _sizej: usize,
        _dxy: F,
    ) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        (sx * dx + sy * dy) / (sx + sy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn average_combines_weighted() {
        let l = AverageLinkage;
        assert_eq!(l.combine(2, 1.0, 1, 3.0, 0, 0.0), 5.0 / 3.0);
    }

    #[test]
    fn agnes_with_average_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, AverageLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn average_linkage_works_with_f32() {
        let l = AverageLinkage;
        // ensure the generic parameter can be f32
        let r: f32 = l.combine(2, 1.0_f32, 1, 3.0_f32, 0, 0.0_f32);
        assert!((r - (5.0_f32 / 3.0)).abs() < f32::EPSILON);
    }
}
