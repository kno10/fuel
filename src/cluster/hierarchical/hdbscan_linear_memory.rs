use crate::DataAccess;

use super::hdbscan_common::{
    HdbscanHierarchy, WeightedEdge, compute_core_distances, edges_to_merge_history,
    mutual_reachability_distance,
};

/// Perform HDBSCAN linear-memory clustering via Prim's MST construction on
/// mutual reachability distances.
#[must_use]
pub fn hdbscan_linear_memory<D: DataAccess>(data: &D, min_points: usize) -> HdbscanHierarchy {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances = compute_core_distances(data, min_points);
    if n == 1 {
        return HdbscanHierarchy::new(Vec::new(), core_distances);
    }

    let mut in_tree = vec![false; n];
    let mut best = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];

    in_tree[0] = true;
    for v in 1..n {
        best[v] = mutual_reachability_distance(data, &core_distances, 0, v);
        parent[v] = 0;
    }

    let mut edges = Vec::with_capacity(n - 1);
    for _ in 1..n {
        let mut next = usize::MAX;
        let mut next_weight = f64::INFINITY;

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
            let candidate = mutual_reachability_distance(data, &core_distances, next, v);
            if candidate < best[v] || (candidate == best[v] && next < parent[v]) {
                best[v] = candidate;
                parent[v] = next;
            }
        }
    }

    let merges = edges_to_merge_history(n, &mut edges);
    HdbscanHierarchy::new(merges, core_distances)
}

#[cfg(test)]
mod tests {
    use crate::{EuclideanDistance, MatrixDataAccess};

    use super::hdbscan_linear_memory;
    use crate::cluster::hierarchical::slink_hdbscan_linear_memory;

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

        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mst = hdbscan_linear_memory(&data, 2);
        let slink = slink_hdbscan_linear_memory(&data, 2);

        assert_eq!(mst.core_distances, slink.core_distances);
        assert_eq!(mst.merges, slink.merges);
    }
}
