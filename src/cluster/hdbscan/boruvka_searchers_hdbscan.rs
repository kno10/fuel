use std::cmp::Ordering;

use super::hdbscan_common::{
    HdbscanHierarchy, SameComponentFilter, compute_core_distances_tree,
    mutual_reachability_distance,
};
use crate::api::{DistanceData, PrioritySearcher, PrioritySearcherFactory, VectorData};
use crate::cluster::hierarchical::common::UnionFind;
use crate::cluster::hierarchical::search_single_link_common::ClusterBuilder;
use crate::{CandidateHeap, DistPair, DistanceSearch, Float, IndexQuery, KnnSearch};

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

/// Boruvka-style heap-of-searchers HDBSCAN MST (index accelerated).
#[must_use]
pub fn boruvka_searchers_hdbscan<'a, S, D, F>(
    tree: &'a S, data: &'a D, min_points: usize,
) -> HdbscanHierarchy<F>
where
    F: Float + 'a,
    D: DistanceData<F> + VectorData<F> + ?Sized + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
    S: KnnSearch<F, D::Query<'a>>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances = compute_core_distances_tree(tree, data, min_points);
    if n == 1 {
        return HdbscanHierarchy::new(Vec::new(), core_distances);
    }

    let mut uf = UnionFind::new(n);
    let mut neighbor_heaps: Vec<Option<CandidateHeap<F>>> =
        (0..n).map(|_| Some(CandidateHeap::new())).collect();
    let mut searchers: Vec<Option<S::Searcher<'a>>> = (0..n).map(|_| None).collect();
    let mut node_cluster = vec![u32::MAX; n];

    // Initialization: use unfiltered search (matching heap_of_searchers) so
    // that the VP-tree traversal order - and thus the candidate set in each
    // heap - is identical across algorithms.
    let mut query = data.query();
    for a in 0..n {
        let mut searcher = tree.priority_searcher();
        query.set_index(a);
        initialize_neighbors(&query, &mut searcher, a, &mut neighbor_heaps[a], &core_distances);
        if neighbor_heaps[a].as_ref().is_some_and(|heap| !heap.is_empty()) {
            searchers[a] = Some(searcher);
        } else {
            neighbor_heaps[a] = None;
        }
    }

    let max_edges = n - 1;
    let mut edges = Vec::with_capacity(max_edges);

    while edges.len() < max_edges {
        // Purge same-component entries and refill as needed.
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
            // MRD-aware lower bound: MRD >= max(core_a, actual_dist), so
            // any candidate must have MRD >= max(core_a, all_lower_bound).
            let effective_lb = core_distances[a]
                .max(searchers[a].as_ref().map_or(F::infinity(), |s| s.all_lower_bound()));
            let needs_refill = heap.peek().is_none_or(|next| next.distance > effective_lb);
            if needs_refill && let Some(ref mut searcher) = searchers[a] {
                query.set_index(a);
                refill_neighbors(
                    &query,
                    searcher,
                    &mut uf,
                    a,
                    heap,
                    &core_distances,
                    &mut node_cluster,
                );
            }
            if heap.is_empty() {
                neighbor_heaps[a] = None;
                searchers[a] = None;
            }
        }

        // Find the cheapest outgoing edge for each component.
        let mut best_point = vec![None; n];
        let mut best_dist = vec![F::infinity(); n];
        for (a, heap_opt) in neighbor_heaps.iter().enumerate() {
            let Some(heap) = heap_opt.as_ref() else {
                continue;
            };
            let Some(top) = heap.peek() else {
                continue;
            };
            let ca = uf.find(a);
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
        candidates.sort_by(|x, y| {
            x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then_with(|| x.1.cmp(&y.1))
        });

        for (dist, a) in candidates {
            if edges.len() == max_edges {
                break;
            }
            let Some(heap) = neighbor_heaps[a].as_mut() else {
                continue;
            };
            let Some(top) = heap.peek() else {
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

    // Build the dendrogram from the sorted edges via ClusterBuilder,
    // matching the merge-history construction used by the other variants.
    debug_assert_eq!(
        edges.len(),
        max_edges,
        "Boruvka MST incomplete: got {} edges, expected {}",
        edges.len(),
        max_edges
    );
    let mut builder = ClusterBuilder::<F>::new(n);
    edges.sort();
    for edge in edges {
        builder.merge_points(edge.a, edge.b, edge.dist);
    }

    HdbscanHierarchy::new(builder.into_history(), core_distances)
}

/// Unfiltered initialization matching heap_of_searchers_hdbscan.
///
/// Using an unfiltered search ensures the VP-tree traversal order (and thus
/// the set of candidates accumulated in the heap) is identical to the other
/// HDBSCAN variants.
fn initialize_neighbors<F: Float, Q, S>(
    query: &Q, searcher: &mut S, query_index: usize, heap: &mut Option<CandidateHeap<F>>,
    core_distances: &[F],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let Some(heap) = heap.as_mut() else {
        return;
    };
    let cd = core_distances[query_index];
    let mut threshold = F::infinity();
    while cd.max(searcher.all_lower_bound()) < threshold {
        let Some(cand) = searcher.next(query) else {
            break;
        };
        if cand.index == query_index {
            continue; // skip self
        }
        let b = cand.index;
        let dist = cd.max(core_distances[b]).max(cand.distance);
        heap.push(DistPair::new(dist, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

fn refill_neighbors<F: Float, Q, S>(
    data: &Q, searcher: &mut S, uf: &mut UnionFind, a: usize, heap: &mut CandidateHeap<F>,
    core_distances: &[F], node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let cd = core_distances[a];
    let mut threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    while cd.max(searcher.all_lower_bound()) < threshold {
        let Some(cand) = searcher
            .next_with_filter(data, &mut SameComponentFilter { uf, query_index: a, node_cluster })
        else {
            break;
        };
        let b = cand.index;
        let dist = mutual_reachability_distance(core_distances, a, b, cand.distance);
        heap.push(DistPair::new(dist, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::boruvka_searchers_hdbscan;
    use crate::TableWithDistance;
    use crate::cluster::hdbscan::hdbscan_prim;
    use crate::distance::Euclidean;
    use crate::vptree::VPTree;

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
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(23);
        let tree = VPTree::<f64>::new(&data, 3, &mut rng);

        let expected = hdbscan_prim(&data, 2);
        let got = boruvka_searchers_hdbscan(&tree, &data, 2);
        assert_eq!(got, expected);
    }

    #[test]
    fn boruvka_searchers_hdbscan_matches_prim_random_200() {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let dim = 5;
        let points: Vec<Vec<f64>> =
            (0..n).map(|_| (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect()).collect();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree = VPTree::<f64>::new(&data, n, &mut rng);

        for min_pts in [2, 5, 10] {
            let expected = hdbscan_prim(&data, min_pts);
            let got = boruvka_searchers_hdbscan(&tree, &data, min_pts);
            assert_eq!(
                got.merges.len(),
                expected.merges.len(),
                "Different number of merges for min_pts={min_pts}"
            );
            // Compare total MST weight
            let w_expected: f64 = expected.merges.iter().map(|m| m.distance).sum();
            let w_got: f64 = got.merges.iter().map(|m| m.distance).sum();
            assert!(
                (w_expected - w_got).abs() < 1e-10,
                "MST weight mismatch for min_pts={min_pts}: expected={w_expected}, got={w_got}"
            );
            // Compare sorted merge distances (same MST cost structure)
            let mut d_expected: Vec<f64> = expected.merges.iter().map(|m| m.distance).collect();
            let mut d_got: Vec<f64> = got.merges.iter().map(|m| m.distance).collect();
            d_expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
            d_got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            for (i, (e, g)) in d_expected.iter().zip(d_got.iter()).enumerate() {
                assert!(
                    (e - g).abs() < 1e-14,
                    "Sorted distance mismatch at index {i} for min_pts={min_pts}: expected={e}, got={g}"
                );
            }
            // Verify core distances match
            assert_eq!(
                got.core_distances, expected.core_distances,
                "Core distances mismatch for min_pts={min_pts}"
            );
        }
    }

    #[test]
    fn boruvka_searchers_hdbscan_matches_prim_random_2000() {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(42);
        let n = 5000;
        let dim = 8;
        let points: Vec<Vec<f64>> =
            (0..n).map(|_| (0..dim).map(|_| rng.gen_range(0.0..1.0)).collect()).collect();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree = VPTree::<f64>::new(&data, n, &mut rng);

        let min_pts = 10;
        let expected = hdbscan_prim(&data, min_pts);
        let got = boruvka_searchers_hdbscan(&tree, &data, min_pts);
        assert_eq!(
            got.merges.len(),
            expected.merges.len(),
            "Different number of merges for min_pts={min_pts}: got={}, expected={}",
            got.merges.len(),
            expected.merges.len()
        );
    }
}
