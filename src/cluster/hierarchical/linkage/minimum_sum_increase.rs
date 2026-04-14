use crate::cluster::hierarchical::{SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Variant of minimum-sum linkage used by HACAM that later applies an
/// additional "total distance" correction during the merge algorithm.
///
/// The raw distance/prototype computation is identical to
/// [`MinimumSumLinkage`]; the increase is handled by subtracting the
/// component totals stored in the cluster summaries.
pub struct MinimumSumIncreaseLinkage;

impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, (F, idsize)> for MinimumSumIncreaseLinkage {
    fn summarize(data: &D, members: &[idsize]) -> (F, idsize) {
        if members.len() == 1 {
            return (F::zero(), members[0]);
        }
        super::minimum_sum::find_medoid(data, members)
    }

    fn cluster_distance(
        data: &D, summary_a: &(F, idsize), summary_b: &(F, idsize), a: &[idsize], b: &[idsize],
    ) -> (F, (F, idsize)) {
        let (d, proto) = super::minimum_sum::minimum_sum_candidate(data, a, b);
        let mut union = Vec::with_capacity(a.len() + b.len());
        union.extend_from_slice(a);
        union.extend_from_slice(b);
        (d - (summary_a.0 + summary_b.0), (d, proto as idsize))
    }

    fn merged_prototype(summary: &(F, idsize)) -> usize { summary.1 as usize }
}
