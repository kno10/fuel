use std::cmp::Ordering;
use std::collections::BinaryHeap;

use num_traits::Float;
#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand::rngs::StdRng;

use crate::api::{DistanceData, DistanceSearch};
use crate::cluster::hdbscan::hdbscan_common::SameComponentFilter;
#[cfg(test)]
use crate::distance::EuclideanDistance;
use crate::DistPair;
use crate::vptree::{PrioritySearcher, VPTree};

use super::common::{MergeHistory, UnionFind};
use super::search_single_link_common::ClusterBuilder;

#[derive(Debug, Clone, Copy)]
struct Edge<F: Float> {
    a: usize,
    b: usize,
    dist: F,
}

impl<F: Float> PartialEq for Edge<F> {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a && self.b == other.b && self.dist == other.dist
    }
}

impl<F: Float> Eq for Edge<F> {}

impl<F: Float> PartialOrd for Edge<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<F: Float> Ord for Edge<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.a.cmp(&other.a))
            .then_with(|| self.b.cmp(&other.b))
    }
}

/// Boruvka-style heap-of-searchers single-link with VP-tree priority search.
#[must_use]
pub fn boruvka_searchers_single_link<D: DistanceData<F>, F: Float>(
    tree: &VPTree<F>,
    data: &D,
) -> MergeHistory<F> {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::<F>::new(n);
    let mut uf = UnionFind::new(n);
    let mut neighbor_heaps: Vec<Option<BinaryHeap<DistPair<F>>>> =
        (0..n).map(|_| Some(BinaryHeap::new())).collect();
    let mut searchers: Vec<Option<PrioritySearcher<F>>> = (0..n).map(|_| None).collect();
    let mut node_cluster = vec![u32::MAX; n];

    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            neighbor_heaps[a] = None;
            continue;
        }
        let mut searcher = tree.priority_searcher();
        initialize_neighbors(
            &data.search_by_index(a),
            &mut searcher,
            &mut builder,
            &mut uf,
            a,
            &mut neighbor_heaps[a],
            &mut node_cluster,
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
        let mut best_dist = vec![F::infinity(); n];
        for a in 0..n {
            let Some(heap) = neighbor_heaps[a].as_mut() else {
                continue;
            };
            let ca = uf.find(a);
            while let Some(top) = heap.peek().copied() {
                if uf.find(top.index) == ca {
                    heap.pop();
                } else {
                    break;
                }
            }
            let needs_refill = heap.peek().is_none_or(|next| {
                next.distance
                    > searchers[a]
                        .as_ref()
                        .expect("searcher must exist for active heap")
                        .all_lower_bound()
            });
            if needs_refill && let Some(searcher) = searchers[a].as_mut() {
                refill_neighbors(
                    &data.search_by_index(a),
                    searcher,
                    &mut uf,
                    a,
                    heap,
                    &mut node_cluster,
                );
            }
            let Some(top) = heap.peek().copied() else {
                neighbor_heaps[a] = None;
                searchers[a] = None;
                continue;
            };
            if top.distance < best_dist[ca] {
                best_dist[ca] = top.distance;
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
        candidates.sort_by(|x, y| x.partial_cmp(&y).unwrap_or(Ordering::Equal));

        for (dist, a) in candidates {
            if edges.len() == max_edges {
                break;
            }
            let Some(heap) = neighbor_heaps[a].as_mut() else {
                continue;
            };
            let ca = uf.find(a);
            while let Some(top) = heap.peek().copied() {
                if uf.find(top.index) == ca {
                    heap.pop();
                } else {
                    break;
                }
            }
            let Some(top) = heap.peek().copied() else {
                continue;
            };
            if top.distance != dist {
                continue;
            }
            heap.pop();
            let b = top.index;
            if uf.union(a, b) {
                edges.push(Edge { a, b, dist });
            }
        }
    }

    edges.sort();
    for edge in edges {
        if builder.merge_points(edge.a, edge.b, edge.dist).is_some() {
            if builder.merge_count() == n - 1 {
                break;
            }
        }
    }

    builder.into_history()
}

fn initialize_neighbors<D: DistanceSearch<F>, F: Float>(
    data: &D,
    searcher: &mut PrioritySearcher<F>,
    builder: &mut ClusterBuilder<F>,
    uf: &mut UnionFind,
    a: usize,
    heap: &mut Option<BinaryHeap<DistPair<F>>>,
    node_cluster: &mut [u32],
) {
    let Some(heap) = heap.as_mut() else {
        return;
    };
    let mut threshold = F::infinity();
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(
            data,
            &mut SameComponentFilter {
                uf,
                query_index: a,
                node_cluster,
            },
        ) else {
            break;
        };
        let b = cand.index;
        let d = cand.distance;
        if d == F::zero() {
            let _ = uf.union(a, b);
            let _ = builder.merge_points(a, b, F::zero());
            continue;
        }
        heap.push(DistPair::new(d, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

fn refill_neighbors<D: DistanceSearch<F>, F: Float>(
    data: &D,
    searcher: &mut PrioritySearcher<F>,
    uf: &mut UnionFind,
    a: usize,
    heap: &mut BinaryHeap<DistPair<F>>,
    node_cluster: &mut [u32],
) {
    let mut threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(
            data,
            &mut SameComponentFilter {
                uf,
                query_index: a,
                node_cluster,
            },
        ) else {
            break;
        };
        let b = cand.index;
        heap.push(DistPair::new(cand.distance, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

#[cfg(test)]
mod tests {
    use crate::TableWithDistance;
    use crate::data::CondensedDistanceMatrix;

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
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(23);
        let tree = VPTree::new(&data, 3, &mut rng);

        let vec = condensed_abs_1d(&points);
        let cm = CondensedDistanceMatrix::new(&vec, points.len());
        let expected = crate::cluster::hierarchical::slink(&cm);
        let got = boruvka_searchers_single_link(&tree, &data);
        assert_eq!(got, expected);
    }
}
