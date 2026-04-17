use crate::cluster::hierarchical::{SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Distance/prototype for medoid linkage.
///
/// The prototype is the medoid of the union of the two clusters.  The linkage
/// distance is the distance between the two medoids.
/// This method is only implemented as a set-based linkage.
pub struct MedoidLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, idsize> for MedoidLinkage {
    fn summarize(data: &D, members: &[idsize]) -> idsize {
        if members.len() <= 1 { members[0] } else { find_medoid(data, members) }
    }

    fn cluster_distance(
        data: &D, summary_a: &idsize, summary_b: &idsize, a: &[idsize], b: &[idsize],
    ) -> (F, idsize) {
        let dist = F::from(data.distance(*summary_a as usize, *summary_b as usize)).unwrap();
        let mut union = Vec::with_capacity(a.len() + b.len());
        union.extend_from_slice(a);
        union.extend_from_slice(b);
        let proto = find_medoid(data, &union);
        (dist, proto)
    }

    fn merged_prototype(summary: &idsize) -> usize { *summary as usize }
}

fn find_medoid<D: DistanceData<F>, F: Float>(data: &D, cluster: &[idsize]) -> idsize {
    let mut best = (cluster[0], F::infinity());

    for &cand in cluster {
        let mut sum = F::zero();
        for &other in cluster {
            if cand != other {
                sum += F::from(data.distance(cand as usize, other as usize)).unwrap();
                if sum >= best.1 {
                    break;
                }
            }
        }
        if sum < best.1 {
            best = (cand, sum);
        }
    }
    best.0
}
