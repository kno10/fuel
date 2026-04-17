use crate::cluster::hierarchical::{SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Linkage criterion that chooses the candidate minimizing the total
/// distance to all points in the two clusters (the usual "minimum-sum").
///
/// This linkage originates from the HACAM method (hierarchical clustering
/// around medoids). The criterion selects the medoid
/// $z \in X \cup Y$ minimizing $\sum_{p\in X\cup Y} d(z, p)$.
/// This method is only implemented as a set-based linkage.
/// The returned prototype is the index of the chosen medoid.
pub struct MinimumSumLinkage;
impl<D: DistanceData<F>, F: Float> SetLinkage<D, F, idsize> for MinimumSumLinkage {
    fn summarize(data: &D, members: &[idsize]) -> idsize { find_medoid(data, members).1 as idsize }

    fn cluster_distance(
        data: &D, _summary_a: &idsize, _summary_b: &idsize, a: &[idsize], b: &[idsize],
    ) -> (F, idsize) {
        minimum_sum_candidate(data, a, b)
    }

    fn merged_prototype(summary: &idsize) -> usize { *summary as usize }
}

pub(crate) fn minimum_sum_candidate<D: DistanceData<F>, F: Float>(
    data: &D, cx: &[idsize], cy: &[idsize],
) -> (F, idsize) {
    debug_assert!(!cx.is_empty() && !cy.is_empty());
    if cx.len() == 1 && cy.len() == 1 {
        return (data.distance(cx[0] as usize, cy[0] as usize), cx[0]);
    }

    let mut best = (F::infinity(), cx[0]);
    for &cand in cx {
        let mut sum = distance_sum(data, cand, cy, F::zero(), best.0);
        if sum >= best.0 {
            continue;
        }
        sum = distance_sum(data, cand, cx, sum, best.0);
        if sum < best.0 {
            best = (sum, cand);
        }
    }
    for &cand in cy {
        let mut sum = distance_sum(data, cand, cx, F::zero(), best.0);
        if sum >= best.0 {
            continue;
        }
        sum = distance_sum(data, cand, cy, sum, best.0);
        if sum < best.0 {
            best = (sum, cand);
        }
    }

    best
}

pub(crate) fn find_medoid<D: DistanceData<F>, F: Float>(data: &D, cx: &[idsize]) -> (F, idsize) {
    debug_assert!(!cx.is_empty());
    let mut best = (F::infinity(), cx[0]);
    for &cand in cx {
        let sum = distance_sum(data, cand, cx, F::zero(), best.0);
        if sum < best.0 {
            best = (sum, cand);
        }
    }
    best
}

#[inline]
fn distance_sum<D: DistanceData<F>, F: Float>(
    data: &D, cand: idsize, cluster: &[idsize], mut sum: F, min_sum: F,
) -> F {
    for &p in cluster {
        sum += data.distance(cand as usize, p as usize);
        if sum >= min_sum {
            return sum;
        }
    }
    sum
}
