use super::common::MergeHistory;
use crate::Float;
use crate::cluster::hdbscan::hdbscan_common::{WeightedEdge, edges_to_merge_history};
use crate::cluster::optics::OpticsResult;

/// Convert an OPTICS cluster order into a hierarchical merge history.
///
/// This follows the ELKI strategy used by `OPTICSToHierarchical`: connect
/// consecutive points in OPTICS order with edge weight equal to reachability
/// and build a dendrogram by sorting these edges by increasing weight.
#[must_use]
pub fn optics_to_hierarchical<F: Float>(result: &OpticsResult<F>) -> MergeHistory<F> {
    let n = result.ordering.len();
    if n <= 1 {
        return Vec::new();
    }

    assert_eq!(result.reachability.len(), n, "reachability length must match ordering length");

    let mut seen = vec![false; n];
    for &idx in &result.ordering {
        assert!(idx < n, "ordering index out of bounds");
        assert!(!seen[idx], "ordering must be a permutation");
        seen[idx] = true;
    }

    let mut edges = Vec::with_capacity(n - 1);
    for pos in 1..n {
        let left = result.ordering[pos - 1];
        let right = result.ordering[pos];
        edges.push(WeightedEdge::new(left, right, result.reachability[right]));
    }

    edges_to_merge_history(n, &mut edges)
}

#[cfg(test)]
mod tests {
    use super::optics_to_hierarchical;
    use crate::cluster::optics::OpticsResult;

    #[test]
    fn optics_order_converts_to_full_hierarchy() {
        let result = OpticsResult {
            ordering: vec![0, 1, 2, 3],
            reachability: vec![f64::INFINITY, 0.3, 0.1, 0.2],
            core_distance: vec![None; 4],
            predecessor: vec![None; 4],
            labels: vec![-1; 4],
        };

        let merges = optics_to_hierarchical(&result);

        assert_eq!(merges.len(), 3);
        assert_eq!(merges[0].idx1, 1);
        assert_eq!(merges[0].idx2, 2);
        assert!((merges[0].distance - 0.1).abs() < 1e-12);
        assert_eq!(merges[1].idx1, 3);
        assert_eq!(merges[1].idx2, 4);
        assert!((merges[1].distance - 0.2).abs() < 1e-12);
        assert_eq!(merges[2].idx1, 0);
        assert_eq!(merges[2].idx2, 5);
        assert!((merges[2].distance - 0.3).abs() < 1e-12);
        assert_eq!(merges[2].size, 4);
    }
}
