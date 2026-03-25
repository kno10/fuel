use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

/// Hausdorff linkage based on the directed-maximum definition.
pub struct HausdorffLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for HausdorffLinkage {
    fn summarize(_data: &D, _members: &[usize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[usize], b: &[usize],
    ) -> (F, Option<usize>) {
        (hausdorff_distance(data, a, b), None)
    }
}

fn hausdorff_distance<D: DistanceData<F>, F: Float>(data: &D, a: &[usize], b: &[usize]) -> F {
    let mut maxmin_ab = F::zero();
    for &p in a {
        let mut min_dist = F::infinity();
        for &q in b {
            let d = F::from(data.distance(p, q)).unwrap();
            if d < min_dist {
                min_dist = d;
                if min_dist == F::zero() {
                    break;
                }
            }
        }
        if min_dist > maxmin_ab {
            maxmin_ab = min_dist;
        }
    }
    let mut maxmin_ba = F::zero();
    for &p in b {
        let mut min_dist = F::infinity();
        for &q in a {
            let d = F::from(data.distance(p, q)).unwrap();
            if d < min_dist {
                min_dist = d;
                if min_dist == F::zero() {
                    break;
                }
            }
        }
        if min_dist > maxmin_ba {
            maxmin_ba = min_dist;
        }
    }
    maxmin_ab.max(maxmin_ba)
}
