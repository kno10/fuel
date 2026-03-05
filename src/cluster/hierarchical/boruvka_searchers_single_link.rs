use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand::rngs::StdRng;

#[cfg(test)]
use crate::EuclideanDistance;
use crate::{DataAccess, DistanceFunction, MatrixDataAccess, PrioritySearcher, VPTree};

use super::common::MergeHistory;
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

/// Boruvka-style heap-of-searchers single-link with VP-tree priority search.
#[must_use]
pub fn boruvka_searchers_single_link<'t, 'm, 'd, T, DF>(
    tree: &'t VPTree,
    data: &'m MatrixDataAccess<'d, T, DF>,
) -> MergeHistory<f64>
where
    DF: DistanceFunction<T>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");

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
        initialize_neighbors(
            &mut searcher,
            &mut builder,
            &mut uf,
            a,
            &mut neighbor_heaps[a],
        );
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
            &mut builder,
            n,
        );
    }

    edges.sort();
    for edge in edges {
        let _ = builder.merge_points(edge.a, edge.b, edge.dist);
        if builder.merge_count() == n - 1 {
            break;
        }
    }

    builder.into_history()
}

fn initialize_neighbors<T, DF>(
    searcher: &mut PrioritySearcher<'_, IndexedQueryData<'_, '_, T, DF>, f64>,
    builder: &mut ClusterBuilder,
    uf: &mut UnionFind,
    a: usize,
    heap: &mut Option<BinaryHeap<Neighbor>>,
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
        let d = cand.distance();
        if d == 0.0 {
            let _ = uf.union(a, b);
            let _ = builder.merge_points(a, b, 0.0);
            continue;
        }
        heap.push(Neighbor { dist: d, index: b });
        threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    }
}

fn refill_neighbors<T, DF>(
    searcher: &mut PrioritySearcher<'_, IndexedQueryData<'_, '_, T, DF>, f64>,
    uf: &mut UnionFind,
    a: usize,
    heap: &mut BinaryHeap<Neighbor>,
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
        heap.push(Neighbor {
            dist: cand.distance(),
            index: b,
        });
        threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    }
}

fn poll_searchers<'t, 'm, 'd, T, DF>(
    uf: &mut UnionFind,
    heaps: &mut [Option<BinaryHeap<Neighbor>>],
    searchers: &mut [Option<PrioritySearcher<'t, IndexedQueryData<'m, 'd, T, DF>, f64>>],
    _builder: &mut ClusterBuilder,
    n: usize,
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
                refill_neighbors(searcher, uf, a, heap);
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
    use crate::cluster::hierarchical::slink;

    use super::*;

    fn condensed_abs_1d(points: &[Vec<f64>]) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 1..points.len() {
            for j in 0..i {
                out.push((points[i][0] - points[j][0]).abs());
            }
        }
        out
    }

    #[test]
    fn boruvka_searchers_matches_slink_on_unique_1d_distances() {
        let points = vec![vec![0.0], vec![1.1], vec![3.7], vec![10.2], vec![20.5]];
        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(23);
        let tree = VPTree::new(&data, 3, &mut rng);

        let expected = slink(&condensed_abs_1d(&points), points.len());
        let got = boruvka_searchers_single_link(&tree, &data);
        assert_eq!(got, expected);
    }
}
