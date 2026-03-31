/// Median linkage (WPGMC).
#[derive(Clone, Copy, Default, Debug)]
pub struct MedianLinkage;

use super::{GeometricLinkage, Linkage};
use crate::Float;

impl<F: Float> Linkage<F> for MedianLinkage {
    fn combine(&self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, dxy: F) -> F {
        F::from(0.5).unwrap() * (dx + dy) - F::from(0.25).unwrap() * dxy
    }
}

impl<F: Float> GeometricLinkage<F> for MedianLinkage {
    fn merge(&self, x: &[F], _sizex: usize, y: &[F], _sizey: usize) -> Vec<F> {
        let half = F::from(0.5).unwrap();
        x.iter().zip(y.iter()).map(|(&xi, &yi)| half * (xi + yi)).collect()
    }

    fn linkage(&self, x: &[F], _sizex: usize, y: &[F], _sizey: usize) -> F {
        let mut total = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let d = *xi - *yi;
            total += d * d;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn median_combine_behaviour() {
        let m = MedianLinkage;
        assert_eq!(m.combine(1, 1.0, 1, 3.0, 0, 1.0), 0.5 * (1.0 + 3.0) - 0.25);
    }

    #[test]
    fn agnes_with_median_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, MedianLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn median_f32_compile() {
        let m = MedianLinkage;
        let r: f32 = m.combine(0, 1.0_f32, 0, 3.0_f32, 0, 1.0_f32);
        assert_eq!(r, 0.5_f32 * (1.0 + 3.0) - 0.25_f32 * 1.0);
    }
}
