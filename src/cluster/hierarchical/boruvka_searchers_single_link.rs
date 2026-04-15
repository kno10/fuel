use std::cmp::Ordering;

use crate::api::{DistanceData, PrioritySearcher, PrioritySearcherFactory, SearchFilter};
use crate::cluster::hdbscan::hdbscan_common::SameComponentFilter;
use crate::cluster::hierarchical::MergeHistory;
use crate::cluster::hierarchical::common::UnionFind;
use crate::cluster::hierarchical::search_single_link_common::ClusterBuilder;
use crate::{CandidateHeap, DistPair, DistanceSearch, Float, IndexQuery};

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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
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

/// Boruvka-style heap-of-searchers single-link with priority-search acceleration.
#[must_use]
pub fn boruvka_searchers_single_link<'a, S, D, F>(tree: &'a S, data: &'a D) -> MergeHistory<F>
where
    F: Float + 'a,
    D: DistanceData<F> + ?Sized + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::<F>::new(n);
    let mut uf = UnionFind::new(n);
    let mut neighbor_heaps: Vec<Option<CandidateHeap<F>>> =
        (0..n).map(|_| Some(CandidateHeap::new())).collect();
    let mut searchers: Vec<Option<S::Searcher<'a>>> = (0..n).map(|_| None).collect();
    let mut node_cluster = vec![u32::MAX; n];

    let mut query = data.query();
    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            neighbor_heaps[a] = None;
            continue;
        }
        let mut searcher = tree.priority_searcher();
        query.set_index(a);
        initialize_neighbors(
            &query,
            &mut searcher,
            &mut builder,
            &mut uf,
            a,
            &mut neighbor_heaps[a],
            &mut node_cluster,
        );
        if neighbor_heaps[a].as_ref().is_some_and(|heap| !heap.is_empty()) {
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
            while let Some(top) = heap.peek() {
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
                query.set_index(a);
                refill_neighbors(&query, searcher, &mut uf, a, heap, &mut node_cluster);
            }
            let Some(top) = heap.peek() else {
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
        candidates.sort_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));

        for (dist, a) in candidates {
            if edges.len() == max_edges {
                break;
            }
            let Some(heap) = neighbor_heaps[a].as_mut() else {
                continue;
            };
            let ca = uf.find(a);
            while let Some(top) = heap.peek() {
                if uf.find(top.index) == ca {
                    heap.pop();
                } else {
                    break;
                }
            }
            let Some(top) = heap.peek() else {
                continue;
            };
            if top.distance != dist {
                continue;
            }
            heap.pop();
            let b = top.index;
            if uf.union(a, b).0 {
                edges.push(Edge { a, b, dist });
            }
        }
    }

    edges.sort();
    for edge in edges {
        if builder.merge_points(edge.a, edge.b, edge.dist).is_some()
            && builder.merge_count() == n - 1
        {
            break;
        }
    }

    builder.into_history()
}

fn initialize_neighbors<F, Q, S>(
    query: &Q, searcher: &mut S, builder: &mut ClusterBuilder<F>, uf: &mut UnionFind, a: usize,
    heap: &mut Option<CandidateHeap<F>>, node_cluster: &mut [u32],
) where
    F: Float,
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let Some(heap) = heap.as_mut() else {
        return;
    };
    let mut threshold = F::infinity();
    while searcher.all_lower_bound() < threshold {
        let (b, d) = {
            let mut filter: SameComponentFilter<'_> =
                SameComponentFilter { uf, query_index: a, node_cluster };
            let Some(cand) = searcher.next(query) else {
                break;
            };
            if filter.skip_point(cand.index) {
                continue;
            }
            (cand.index, cand.distance)
        };

        if d == F::zero() {
            let _ = uf.union(a, b);
            let _ = builder.merge_points(a, b, F::zero());
            continue;
        }
        heap.push(DistPair::new(d, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

fn refill_neighbors<F, Q, S>(
    query: &Q, searcher: &mut S, uf: &mut UnionFind, a: usize, heap: &mut CandidateHeap<F>,
    node_cluster: &mut [u32],
) where
    F: Float,
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let mut threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    while searcher.all_lower_bound() < threshold {
        let mut filter: SameComponentFilter<'_> =
            SameComponentFilter { uf, query_index: a, node_cluster };
        let Some(cand) = searcher.next_with_filter(query, &mut filter) else {
            break;
        };
        let b = cand.index;
        heap.push(DistPair::new(cand.distance, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::search::vptree::VPTree;

    #[test]
    fn boruvka_searchers_single_link_regression() {
        test_clustering_table(
            "BoruvkaSearchersSingleLink",
            "single",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let mut rng = StdRng::seed_from_u64(42);
                let tree = VPTree::new(access, 3, &mut rng);
                let history = boruvka_searchers_single_link(&tree, access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
