use crate::cluster::hierarchical::{SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Variant of minimum-sum linkage used by HACAM that later applies an
/// additional "total distance" correction during the merge algorithm.
///
/// This linkage originates from hierarchical clustering around medoids (HACAM).
/// The raw distance/prototype computation is identical to
/// [`crate::cluster::hierarchical::MinimumSumLinkage`]; the merge distance is corrected by subtracting the
/// stored intra-cluster totals, similar to Ward's method.
/// This method is only implemented as a set-based linkage.
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
