use crate::cluster::hierarchical::{SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Hausdorff linkage based on the directed-maximum definition.
///
/// The linkage is the Hausdorff distance between clusters:
/// $\max\{ \max_{x\in X} \min_{y\in Y} d(x,y),\ \max_{y\in Y} \min_{x\in X} d(x,y) \}$.
/// This method can produce inversions and is only implemented as a set-based
/// linkage.
pub struct HausdorffLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, ()> for HausdorffLinkage {
    fn can_produce_inversions(&self) -> bool { true }

    fn summarize(_data: &D, _members: &[idsize]) {}

    fn cluster_distance(
        data: &D, _summary_a: &(), _summary_b: &(), a: &[idsize], b: &[idsize],
    ) -> (F, ()) {
        (hausdorff_distance(data, a, b), ())
    }
}

fn hausdorff_distance<D: DistanceData<F>, F: Float>(data: &D, a: &[idsize], b: &[idsize]) -> F {
    let mut maxmin_ab = F::zero();
    for &p in a {
        let mut min_dist = F::infinity();
        for &q in b {
            let d = F::from(data.distance(p as usize, q as usize)).unwrap();
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
            let d = F::from(data.distance(p as usize, q as usize)).unwrap();
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
