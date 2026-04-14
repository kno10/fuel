/// Median linkage (WPGMC).
#[derive(Clone, Copy, Default, Debug)]
pub struct MedianLinkage;

use crate::cluster::hierarchical::{GeometricLinkage, Linkage};
use crate::distance::squared_euclidean_distance;
use crate::{Float, math};

impl<F: Float> Linkage<F> for MedianLinkage {
    fn can_produce_inversions(&self) -> bool { true }

    fn initial(&self, d: F, issquare: bool) -> F { if issquare { d } else { d * d } }

    fn restore(&self, d: F, issquare: bool) -> F { if issquare { d } else { d.sqrt() } }

    fn combine(
        &self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        F::half() * (dx + dy) - F::quarter() * dxy
    }
}

impl<F: Float> GeometricLinkage<F> for MedianLinkage {
    fn merge(
        &self, x: &[F], _sizex: usize, y: &[F], _sizey: usize, _heightx: F, _heighty: F,
    ) -> Vec<F> {
        debug_assert!(x.len() == y.len());
        let mut out = x.to_vec();
        math::axpby(&mut out, F::half(), y, F::half(), x.len());
        out
    }

    fn linkage(
        &self, x: &[F], _sizex: usize, y: &[F], _sizey: usize, _heightx: F, _heighty: F,
    ) -> F {
        squared_euclidean_distance(x, y)
    }
}
