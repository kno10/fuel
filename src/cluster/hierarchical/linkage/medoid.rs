use crate::cluster::hierarchical::SetLinkage;
use crate::{DistanceData, Float};

pub(crate) struct MedoidSummary {
    medoid: usize,
}

/// Distance/prototype for medoid linkage.  The prototype is the medoid of the
/// union of the two clusters.  The linkage computation iterates over all
/// candidates in both clusters and therefore does not need access to any
/// previously stored medoid value beyond the current proxies.
pub struct MedoidLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, MedoidSummary> for MedoidLinkage {
    fn summarize(_data: &D, members: &[usize]) -> MedoidSummary {
        MedoidSummary { medoid: members[0] }
    }

    fn cluster_distance(
        data: &D, summary_a: &MedoidSummary, summary_b: &MedoidSummary, a: &[usize], b: &[usize],
    ) -> (F, Option<usize>) {
        let dist = F::from(data.distance(summary_a.medoid, summary_b.medoid)).unwrap();
        let mut union = Vec::with_capacity(a.len() + b.len());
        union.extend_from_slice(a);
        union.extend_from_slice(b);
        let proto = find_medoid(data, &union);
        (dist, Some(proto))
    }

    fn merge_summary(
        dest: &mut MedoidSummary, _source: MedoidSummary, prototype: Option<usize>, _distance: F,
    ) {
        if let Some(proto) = prototype {
            dest.medoid = proto;
        }
    }
}

fn find_medoid<D: DistanceData<F>, F: Float>(data: &D, cluster: &[usize]) -> usize {
    let mut best = cluster[0];
    let mut min_sum = F::infinity();

    for &cand in cluster {
        let mut sum = F::zero();
        for &other in cluster {
            if cand != other {
                sum += F::from(data.distance(cand, other)).unwrap();
                if sum >= min_sum {
                    break;
                }
            }
        }
        if sum < min_sum {
            min_sum = sum;
            best = cand;
        }
    }
    best
}
