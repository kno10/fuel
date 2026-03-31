use std::collections::HashMap;

use rayon::prelude::*;

use crate::api::{DistanceData, RangeSearch};
use crate::cluster::dbscan::NOISE;
use crate::{Float, IndexQuery};

/// Run DBSCAN in parallel by finding connected components of core points.
///
/// The algorithm first identifies all core points whose epsilon-ball contains at
/// least `min_points`. It then builds a union-find over the core points by
/// connecting every pair of neighboring cores. After the union-find has merged
/// connected components, every point is assigned either the cluster label of a
/// nearby core point or `NOISE`.
pub fn parallel_dbscan<'a, S, D, F>(tree: &S, data: &'a D, eps: F, min_points: usize) -> Vec<isize>
where
    F: Float + Sync,
    D: DistanceData<F> + Sync + 'a,
    D::Query<'a>: Send,
    S: RangeSearch<F, D::Query<'a>> + Sync,
{
    assert!(eps >= F::zero(), "eps must be non-negative");
    assert!(min_points > 0, "min_points must be greater than 0");

    let size = data.len();
    if size == 0 {
        return Vec::new();
    }

    // Identify core points in parallel.
    let is_core: Vec<bool> = (0..size)
        .into_par_iter()
        .map_init(
            || data.query(),
            |query, idx| {
                query.set_index(idx);
                let mut neighbors = 0usize;
                for _ in tree.search_range(query, eps) {
                    neighbors += 1;
                }
                neighbors >= min_points
            },
        )
        .collect();

    let mut core_ids = vec![None; size];
    let mut cores = Vec::new();
    for (idx, &core) in is_core.iter().enumerate() {
        if core {
            core_ids[idx] = Some(cores.len());
            cores.push(idx);
        }
    }

    if cores.is_empty() {
        return vec![NOISE; size];
    }

    let num_cores = cores.len();
    let mut union_find = UnionFind::new(num_cores);
    let core_id_by_point = core_ids;

    let mut query = data.query();
    for (core_idx, &point_idx) in cores.iter().enumerate() {
        query.set_index(point_idx);
        for pair in tree.search_range(&query, eps) {
            if let Some(neighbor_core_idx) = core_id_by_point[pair.index] {
                union_find.union(core_idx, neighbor_core_idx);
            }
        }
    }

    let mut root_to_label = HashMap::with_capacity(num_cores);
    let mut next_label: isize = 0;
    let mut cluster_label_by_core = vec![NOISE; num_cores];
    for (core_idx, slot) in cluster_label_by_core.iter_mut().enumerate().take(num_cores) {
        let root = union_find.find(core_idx);
        let label = *root_to_label.entry(root).or_insert_with(|| {
            let label = next_label;
            next_label += 1;
            label
        });
        *slot = label;
    }

    // Assign every point to its cluster label (or noise).
    (0..size)
        .into_par_iter()
        .map_init(
            || data.query(),
            |query, idx| {
                if let Some(core_idx) = core_id_by_point[idx] {
                    return cluster_label_by_core[core_idx];
                }
                let mut assigned = NOISE;
                query.set_index(idx);
                for pair in tree.search_range(query, eps) {
                    if assigned != NOISE {
                        break;
                    }
                    if let Some(neighbor_core_idx) = core_id_by_point[pair.index] {
                        assigned = cluster_label_by_core[neighbor_core_idx];
                    }
                }
                assigned
            },
        )
        .collect()
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self { Self { parent: (0..n).collect(), size: vec![1; n] } }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        let root = x;
        let mut node = x;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    fn union(&mut self, a: usize, b: usize) -> usize {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return ra;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        ra
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    fn build_tree<'a>(
        points: &'a [Vec<f64>],
    ) -> (TableWithDistance<'a, f64, Vec<f64>, Euclidean, f64>, VPTree<f64>) {
        let data = TableWithDistance::with_distance(points, Euclidean);
        let mut rng = StdRng::seed_from_u64(7);
        let tree = VPTree::new(&data, 2, &mut rng);
        (data, tree)
    }

    #[test]
    fn matches_sequential_two_clusters() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![5.0, 5.0],
        ];
        let (data, tree) = build_tree(&points);
        let parallel_labels = parallel_dbscan(&tree, &data, 0.25, 3);
        use crate::cluster::dbscan::dbscan;
        let sequential_labels = dbscan(&tree, &data, 0.25, 3);
        assert_eq!(parallel_labels, sequential_labels);
    }

    #[test]
    fn matches_expected_noise_cases() {
        let points =
            vec![vec![0.0], vec![2.0], vec![3.0], vec![4.0], vec![6.0], vec![8.0], vec![10.0]];
        let (data, tree) = build_tree(&points);
        let expected_cases = [
            (1, vec![0, 1, 1, 1, 2, 3, 4]),
            (2, vec![NOISE, 0, 0, 0, NOISE, NOISE, NOISE]),
            (3, vec![NOISE, 0, 0, 0, NOISE, NOISE, NOISE]),
            (4, vec![NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, NOISE]),
        ];
        for (min_points, expected) in expected_cases {
            let parallel_labels = parallel_dbscan(&tree, &data, 1.0, min_points);
            assert_eq!(parallel_labels, expected);
        }
    }
}
