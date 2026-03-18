/// Minimum variance linkage (MIVAR).
#[derive(Clone, Copy, Default, Debug)]
pub struct MinimumVarianceLinkage;

use super::Linkage;
use crate::DistanceData;
use crate::cluster::hierarchical::SetLinkage;
use num_traits::Float;

impl<F: Float> Linkage<F> for MinimumVarianceLinkage {
    fn initial(&self, d: F, issquare: bool) -> F {
        F::from(0.25).unwrap() * if issquare { d } else { d * d }
    }

    fn restore(&self, d: F, issquare: bool) -> F {
        if issquare {
            F::from(4.0).unwrap() * d
        } else {
            (F::from(4.0).unwrap() * d).sqrt()
        }
    }

    fn combine(&self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F) -> F {
        let xj = F::from(sizex + sizej).unwrap();
        let yj = F::from(sizey + sizej).unwrap();
        let n = F::from(sizex + sizey + sizej).unwrap();
        (xj * xj * dx + yj * yj * dy - F::from(sizej * (sizex + sizey)).unwrap() * dxy) / (n * n)
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for MinimumVarianceLinkage {
    fn summarize(_data: &D, _members: &[usize]) {}

    fn cluster_distance(
        data: &D,
        _summary_a: &(),
        _summary_b: &(),
        a: &[usize],
        b: &[usize],
    ) -> (F, Option<usize>) {
        let mut sum = F::zero();
        let mut count = 0usize;
        for &i in a {
            for &j in b {
                sum = sum + F::from(data.distance(i, j)).unwrap();
                count += 1;
            }
        }
        (sum / F::from(count).unwrap(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn minimum_variance_initial_restore() {
        let m = MinimumVarianceLinkage;
        assert_eq!(m.initial(2.0, false), 1.0);
        assert_eq!(m.restore(1.0, false), 2.0);
    }

    #[test]
    fn agnes_with_minimum_variance_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, MinimumVarianceLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn minimum_variance_f32_compile() {
        let m = MinimumVarianceLinkage;
        let r: f32 = m.initial(2.0_f32, false);
        assert_eq!(r, 1.0_f32);
        let restored: f32 = m.restore(1.0_f32, false);
        assert_eq!(restored, 2.0_f32);
    }
}
