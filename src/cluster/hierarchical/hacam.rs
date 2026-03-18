use crate::DistanceData;
use num_traits::Float;

use super::common::PrototypeMergeHistory;
use crate::cluster::hierarchical::SetLinkage;
use crate::cluster::hierarchical::set_anderberg::set_anderberg_common;

/// Hierarchical Agglomerative Clustering Around Medoids (HACAM).
#[must_use]
pub fn hacam<D, F, L, S>(data: &D, linkage: L) -> PrototypeMergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let _ = linkage;
    // HACAM differs from the plain Anderberg heuristic only in the condition
    // used to decide whether to recompute the best neighbour for the surviving
    // cluster `y`.  Delegate the heavy lifting to the common helper.
    set_anderberg_common::<D, L, F, S, _>(data, |y, x, besti| besti[y] == x)
    // FIXME: check if the same heuristic can always be used?
}

// the rest of the file previously contained an `update_matrices` function and
// tests; both are unnecessary now that the implementation is shared.

#[cfg(test)]
mod tests {
    use crate::TableWithDistance;
    use crate::cluster::hierarchical::test_utils::ScalarDistance;

    use super::hacam;
    use crate::cluster::hierarchical::linkage::minimum_sum::MinimumSumLinkage;
    use crate::cluster::hierarchical::linkage::minimum_sum_increase::MinimumSumIncreaseLinkage;

    #[test]
    fn hacam_variants_return_valid_histories() {
        let data = TableWithDistance::with_distance(&[0.0, 0.5, 2.0, 3.0, 8.0], ScalarDistance);
        let a = hacam(&data, MinimumSumLinkage);
        let b = hacam(&data, MinimumSumIncreaseLinkage);
        assert_eq!(a.len(), 4);
        assert_eq!(b.len(), 4);
        assert_eq!(a.last().expect("non-empty").size, 5);
        assert_eq!(b.last().expect("non-empty").size, 5);
    }
}
