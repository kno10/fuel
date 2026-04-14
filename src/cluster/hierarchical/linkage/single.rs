/// Single-linkage criterion (minimum distance).
#[derive(Clone, Copy, Default, Debug)]
pub struct SingleLinkage;

use crate::cluster::hierarchical::{Linkage, SetLinkage, idsize};
use crate::{DistanceData, Float};

impl<F: Float> Linkage<F> for SingleLinkage {
    fn combine(
        &self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, _dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        dx.min(dy)
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for SingleLinkage {
    fn summarize(_data: &D, _members: &[idsize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[idsize], b: &[idsize],
    ) -> (F, ()) {
        let mut best = F::infinity();
        for &i in a {
            for &j in b {
                let d = F::from(data.distance(i as usize, j as usize)).unwrap();
                if d < best {
                    best = d;
                }
            }
        }
        (best, ())
    }
}
