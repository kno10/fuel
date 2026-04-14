/// Complete-linkage criterion (maximum distance).
#[derive(Clone, Copy, Default, Debug)]
pub struct CompleteLinkage;

use crate::cluster::hierarchical::{Linkage, SetLinkage, idsize};
use crate::{DistanceData, Float};

impl<F: Float> Linkage<F> for CompleteLinkage {
    fn combine(
        &self, _sizex: usize, dx: F, _sizey: usize, dy: F, _sizej: usize, _dxy: F, _heightx: F,
        _heighty: F, _heightj: F,
    ) -> F {
        dx.max(dy)
    }
}

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, idsize> for CompleteLinkage {
    fn summarize(_data: &D, members: &[idsize]) -> idsize { members[0] }

    fn cluster_distance(
        data: &D, _summary_a: &idsize, _summary_b: &idsize, a: &[idsize], b: &[idsize],
    ) -> (F, idsize) {
        let mut best = F::zero();
        let mut proto = a[0];
        for &i in a {
            for &j in b {
                let d = F::from(data.distance(i as usize, j as usize)).unwrap();
                if d > best {
                    best = d;
                    proto = j;
                }
            }
        }
        (best, proto)
    }

    fn merged_prototype(summary: &idsize) -> usize { *summary as usize }
}
