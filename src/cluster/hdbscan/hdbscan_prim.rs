use num_traits::Float;

use super::hdbscan_common::{
    HdbscanHierarchy, WeightedEdge, compute_core_distances, edges_to_merge_history,
    mutual_reachability_distance,
};
use crate::DistanceData;

/// Perform HDBSCAN clustering via Prim's MST construction on
/// mutual reachability distances.  This was formerly exposed as
/// `hdbscan_linear_memory` but is now renamed to emphasise the underlying
/// Prim/MST approach.
#[must_use]
/// generic float version of prim-based HDBSCAN
pub fn hdbscan_prim<F: Float, D: DistanceData<F>>(
    data: &D,
    min_points: usize,
) -> HdbscanHierarchy<F> {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances: Vec<F> = compute_core_distances(data, min_points);

    let mut in_tree = vec![false; n];
    let mut best = vec![F::infinity(); n];
    let mut parent = vec![usize::MAX; n];

    in_tree[0] = true;
    for v in 1..n {
        best[v] = mutual_reachability_distance(&core_distances, 0, v, data.distance(0, v));
        parent[v] = 0;
    }

    let mut edges = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let mut next = usize::MAX;
        let mut next_weight = F::infinity();

        for v in 0..n {
            if in_tree[v] {
                continue;
            }
            if best[v] < next_weight {
                next = v;
                next_weight = best[v];
            }
        }

        assert!(next != usize::MAX, "MST construction failed");
        in_tree[next] = true;
        edges.push(WeightedEdge::new(next, parent[next], next_weight));

        for v in 0..n {
            if in_tree[v] {
                continue;
            }
            let candidate: F =
                mutual_reachability_distance(&core_distances, next, v, data.distance(next, v));
            if candidate < best[v] || (candidate == best[v] && next < parent[v]) {
                best[v] = candidate;
                parent[v] = next;
            }
        }
    }

    HdbscanHierarchy::new(edges_to_merge_history::<F>(n, &mut edges), core_distances)
}

#[cfg(test)]
mod tests {
    use crate::TableWithDistance;
    use crate::cluster::hdbscan::HdbscanHierarchy;
    use crate::distance::EuclideanDistance;

    use super::hdbscan_prim;
    use crate::cluster::hdbscan::slink_hdbscan;

    #[test]
    fn prim_hdbscan_matches_slink_hdbscan() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 1.2],
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mst: HdbscanHierarchy<f64> = hdbscan_prim(&data, 2);
        let slink: HdbscanHierarchy<f64> = slink_hdbscan(&data, 2);

        assert_eq!(mst.core_distances, slink.core_distances);
        assert_eq!(mst.merges, slink.merges);
    }
}
