use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{DataAccess, DistanceFunction, MatrixDataAccess, PrioritySearcher, VPTree};

use super::hdbscan_common::{
    HdbscanHierarchy, compute_core_distances, mutual_reachability_distance_from_distance,
};
use super::search_single_link_common::{ClusterBuilder, IndexedQueryData};

#[derive(Debug, Clone, Copy)]
struct Neighbor {
    dist: f64,
    index: usize,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.dist == other.dist
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    a: usize,
    b: usize,
    dist: f64,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.dist == other.dist
    }
}

impl Eq for Edge {}

impl PartialOrd for Edge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Edge {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.a.cmp(&other.a))
            .then_with(|| self.b.cmp(&other.b))
    }
}

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        let p = self.parent[x];
        if p != x {
            let r = self.find(p);
            self.parent[x] = r;
            r
        } else {
            x
        }
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        true
    }
}

/// Boruvka-style heap-of-searchers HDBSCAN MST (index accelerated).
#[must_use]
pub fn boruvka_searchers_hdbscan<'t, 'm, 'd, T, DF>(
    tree: &'t VPTree,
    data: &'m MatrixDataAccess<'d, T, DF>,
    min_points: usize,
) -> HdbscanHierarchy
where
    DF: DistanceFunction<T>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances = compute_core_distances(data, min_points);
    if n == 1 {
        return HdbscanHierarchy::new(Vec::new(), core_distances);
    }

    let mut builder = ClusterBuilder::new(n);
    let mut uf = UnionFind::new(n);
    let mut neighbor_heaps: Vec<Option<BinaryHeap<Neighbor>>> =
        (0..n).map(|_| Some(BinaryHeap::new())).collect();
    let mut searchers: Vec<Option<PrioritySearcher<'t, IndexedQueryData<'m, 'd, T, DF>, f64>>> =
        (0..n).map(|_| None).collect();

    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            neighbor_heaps[a] = None;
            continue;
        }
        let mut searcher = tree.priority_searcher(IndexedQueryData {
            data,
            query_index: a,
        });
        initialize_neighbors(&mut searcher, a, &mut neighbor_heaps[a], &core_distances);
        if neighbor_heaps[a]
            .as_ref()
            .is_some_and(|heap| !heap.is_empty())
        {
            searchers[a] = Some(searcher);
        } else {
            neighbor_heaps[a] = None;
        }
    }

    let max_edges = n.saturating_sub(builder.merge_count() + 1);
    let mut edges = Vec::with_capacity(max_edges);

    while edges.len() < max_edges {
        let mut best_point = vec![None; n];
        let mut best_dist = vec![f64::INFINITY; n];
        for a in 0..n {
            let Some(heap) = neighbor_heaps[a].as_ref() else {
                continue;
            };
            let Some(top) = heap.peek() else {
                continue;
            };
            let ca = uf.find(a);
            if top.dist < best_dist[ca] {
                best_dist[ca] = top.dist;
                best_point[ca] = Some(a);
            }
        }

        let mut candidates = Vec::new();
        for c in 0..n {
            if let Some(a) = best_point[c] {
                candidates.push((best_dist[c], a));
            }
        }
        if candidates.is_empty() {
            break;
        }
        candidates.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal));

        for (dist, a) in candidates {
            if edges.len() == max_edges {
                break;
            }
            let Some(heap) = neighbor_heaps[a].as_mut() else {
                continue;
            };
            let Some(top) = heap.peek().copied() else {
                continue;
            };
            if top.dist.to_bits() != dist.to_bits() {
                continue;
            }
            heap.pop();
            let b = top.index;
            if uf.union(a, b) {
                edges.push(Edge { a, b, dist });
            }
        }

        poll_searchers(
            &mut uf,
            &mut neighbor_heaps,
            &mut searchers,
            n,
            &core_distances,
        );
    }

    edges.sort();
    for edge in edges {
        let _ = builder.merge_points(edge.a, edge.b, edge.dist);
        if builder.merge_count() == n - 1 {
            break;
        }
    }

    HdbscanHierarchy::new(builder.into_history(), core_distances)
}

fn initialize_neighbors<T, DF>(
    searcher: &mut PrioritySearcher<'_, IndexedQueryData<'_, '_, T, DF>, f64>,
    a: usize,
    heap: &mut Option<BinaryHeap<Neighbor>>,
    core_distances: &[f64],
) where
    DF: DistanceFunction<T>,
{
    let Some(heap) = heap.as_mut() else {
        return;
    };
    let mut threshold = f64::INFINITY;
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next() else {
            break;
        };
        let b = cand.index();
        if a == b {
            continue;
        }
        let dist =
            mutual_reachability_distance_from_distance(core_distances, a, b, cand.distance());
        heap.push(Neighbor { dist, index: b });
        threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    }
}

fn refill_neighbors<T, DF>(
    searcher: &mut PrioritySearcher<'_, IndexedQueryData<'_, '_, T, DF>, f64>,
    uf: &mut UnionFind,
    a: usize,
    heap: &mut BinaryHeap<Neighbor>,
    core_distances: &[f64],
) where
    DF: DistanceFunction<T>,
{
    let mut threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next() else {
            break;
        };
        let b = cand.index();
        if a == b || uf.find(b) == uf.find(a) {
            continue;
        }
        let dist =
            mutual_reachability_distance_from_distance(core_distances, a, b, cand.distance());
        heap.push(Neighbor { dist, index: b });
        threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    }
}

fn poll_searchers<'t, 'm, 'd, T, DF>(
    uf: &mut UnionFind,
    heaps: &mut [Option<BinaryHeap<Neighbor>>],
    searchers: &mut [Option<PrioritySearcher<'t, IndexedQueryData<'m, 'd, T, DF>, f64>>],
    n: usize,
    core_distances: &[f64],
) where
    DF: DistanceFunction<T>,
{
    for a in 0..n {
        let Some(heap) = heaps[a].as_mut() else {
            continue;
        };
        if !heap.is_empty() {
            let ca = uf.find(a);
            while let Some(top) = heap.peek().copied() {
                if uf.find(top.index) == ca {
                    heap.pop();
                } else {
                    break;
                }
            }
        }

        let needs_refill = heap.peek().is_none_or(|next| {
            next.dist
                > searchers[a]
                    .as_ref()
                    .expect("searcher must exist for active heap")
                    .all_lower_bound()
        });
        if needs_refill {
            if let Some(searcher) = searchers[a].as_mut() {
                refill_neighbors(searcher, uf, a, heap, core_distances);
            }
        }

        if heap.is_empty() {
            heaps[a] = None;
            searchers[a] = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::hdbscan_linear_memory;
    use crate::{EuclideanDistance, MatrixDataAccess, VPTree};
    use rand::{SeedableRng, rngs::StdRng};

    use super::boruvka_searchers_hdbscan;

    #[test]
    fn boruvka_searchers_hdbscan_matches_prim_mst() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 1.2],
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];
        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(23);
        let tree = VPTree::new(&data, 3, &mut rng);

        let expected = hdbscan_linear_memory(&data, 2);
        let got = boruvka_searchers_hdbscan(&tree, &data, 2);
        assert_eq!(got, expected);
    }
}
