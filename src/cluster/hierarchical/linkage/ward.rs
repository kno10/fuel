/// Ward linkage (minimize increase of sum-of-squares).
///
/// This method is special in that it operates naturally on *squared*
/// distances; therefore both `initial` and `restore` are overridden.  It also
/// implements `GeometricLinkage` to provide centroid merging support.
#[derive(Clone, Copy, Default, Debug)]
pub struct WardLinkage;

use super::GeometricLinkage;
use super::Linkage;
use crate::DistanceData;
use crate::cluster::hierarchical::SetLinkage;
use num_traits::Float;

impl<F: Float> Linkage<F> for WardLinkage {
    fn initial(&self, d: F, issquare: bool) -> F {
        F::from(0.5).unwrap() * if issquare { d } else { d * d }
    }

    fn restore(&self, d: F, issquare: bool) -> F {
        if issquare {
            F::from(2.0).unwrap() * d
        } else {
            (F::from(2.0).unwrap() * d).sqrt()
        }
    }

    fn combine(&self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let sj = F::from(sizej).unwrap();
        ((sx + sj) * dx + (sy + sj) * dy - sj * dxy) / (sx + sy + sj)
    }
}

impl<F: Float> GeometricLinkage<F> for WardLinkage {
    fn merge(&self, x: &[F], sizex: usize, y: &[F], sizey: usize) -> Vec<F> {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        // weighted mean of vectors
        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (sx * xi + sy * yi) / (sx + sy))
            .collect()
    }

    fn linkage(&self, x: &[F], sizex: usize, y: &[F], sizey: usize) -> F {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        F::from(0.5).unwrap() * sx * sy / (sx + sy) * {
            let mut sum = F::zero();
            for (xi, yi) in x.iter().zip(y.iter()) {
                let diff = *xi - *yi;
                sum = sum + diff * diff;
            }
            sum
        }
    }

    fn restore_linkage(&self, d: F, issquare: bool) -> F {
        if issquare {
            F::from(4.0).unwrap() * d
        } else {
            F::from(2.0).unwrap() * d.sqrt()
        }
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for WardLinkage {
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
    fn ward_initial_restore() {
        let l = WardLinkage;
        // initial with non-squared should square
        assert_eq!(l.initial(3.0, false), 4.5);
        assert_eq!(l.restore(4.5, false), (9.0_f64).sqrt());
    }

    #[test]
    fn agnes_with_ward_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, WardLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }

    #[test]
    fn ward_f32_compile() {
        let l = WardLinkage;
        let _ = l.initial(3.0_f32, false);
        let _ = l.restore(4.5_f32, false);
    }
}
