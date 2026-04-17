/// Centroid linkage (UPGMC).
///
/// Uses cluster sizes in the Lance-Williams formula and corresponds to the
/// squared distance between cluster centroids when using Euclidean distance.
/// The recurrence is:
/// $\frac{n_x d_x + n_y d_y - \frac{n_x n_y}{n_x + n_y} d_{xy}}{n_x + n_y}$.
/// This method can produce inversions.
/// It supports stored-matrix algorithms and geometric stored-data approaches.
#[derive(Clone, Copy, Default, Debug)]
pub struct CentroidLinkage;

use crate::Float;
use crate::cluster::hierarchical::{GeometricLinkage, Linkage};
use crate::distance::squared_euclidean_distance;

impl<F: Float> Linkage<F> for CentroidLinkage {
    fn can_produce_inversions(&self) -> bool { true }

    fn initial(&self, d: F, issquare: bool) -> F { if issquare { d } else { d * d } }

    fn restore(&self, d: F, issquare: bool) -> F { if issquare { d } else { d.sqrt() } }

    fn combine(
        &self, sizex: usize, dx: F, sizey: usize, dy: F, _sizej: usize, dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        let tot = F::from(sizex + sizey).unwrap();
        let f = F::one() / tot;
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        (sx * dx + sy * dy - (sx * sy) * f * dxy) * f
    }
}

impl<F: Float> GeometricLinkage<F> for CentroidLinkage {
    fn linkage(
        &self, x: &[F], _sizex: usize, y: &[F], _sizey: usize, _heightx: F, _heighty: F,
    ) -> F {
        squared_euclidean_distance(x, y)
    }
}
