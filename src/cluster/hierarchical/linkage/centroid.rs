/// Centroid linkage (UPGMC).
///
/// Uses cluster sizes in the Lance–Williams formula and corresponds to the
/// squared distance between cluster centroids when using Euclidean distance.
#[derive(Clone, Copy, Default, Debug)]
pub struct CentroidLinkage;

use super::{GeometricLinkage, Linkage};
use crate::Float;

impl<F: Float> Linkage<F> for CentroidLinkage {
    fn combine(&self, sizex: usize, dx: F, sizey: usize, dy: F, _sizej: usize, dxy: F) -> F {
        let tot = F::from(sizex + sizey).unwrap();
        let f = F::one() / tot;
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        (sx * dx + sy * dy - (sx * sy) * f * dxy) * f
    }
}

impl<F: Float> GeometricLinkage<F> for CentroidLinkage {
    fn merge(&self, x: &[F], sizex: usize, y: &[F], sizey: usize) -> Vec<F> {
        let tot = F::from(sizex + sizey).unwrap();
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        x.iter().zip(y.iter()).map(|(&xi, &yi)| (sx * xi + sy * yi) / tot).collect()
    }

    fn linkage(&self, x: &[F], _sizex: usize, y: &[F], _sizey: usize) -> F {
        // squared Euclidean
        let mut total = F::zero();
        for (xi, yi) in x.iter().zip(y.iter()) {
            let d = *xi - *yi;
            total = total + d * d;
        }
        total
    }
}

// SetLinkage is not provided for centroid linkage because the trait is
// defined in terms of pairwise distances only; computing the squared
// distance between centroids requires coordinate information that is not
// available through `DataAccess`.  Consumers who need this behaviour can
// fall back to `agnes` with `CentroidLinkage` directly or implement a
// custom set-based variant if they have access to coordinates.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::agnes;

    #[test]
    fn centroid_combine_behaviour() {
        let l = CentroidLinkage;
        // simple check formula consistency
        let d: f64 = l.combine(2, 1.0_f64, 3, 2.0_f64, 1, 0.5_f64);
        assert!(d.is_finite());
    }

    #[test]
    fn centroid_linkage_f32_compiles() {
        let l = CentroidLinkage;
        let r: f32 = l.combine(1, 1.0_f32, 1, 2.0_f32, 0, 0.5_f32);
        assert!(r.is_finite());
    }

    #[test]
    fn agnes_with_centroid_runs() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let history = agnes(&d, 4, CentroidLinkage, false);
        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().size, 4);
    }
}
