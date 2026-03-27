use crate::api::{DistanceData, PrioritySearcher, PrioritySearcherFactory};
use crate::cluster::hdbscan::hdbscan_common::{HdbscanHierarchy, compute_core_distances_tree};
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{CandidateHeap, DistPair, DistanceSearch, Float, IndexQuery, KnnSearch};

/// Heap-of-searchers HDBSCAN-HS (HSSL-style acceleration with priority-search acceleration).
#[must_use]
pub fn heap_of_searchers_hdbscan<'a, S, D, F>(
    tree: &'a S, data: &'a D, min_points: usize,
) -> HdbscanHierarchy<F>
where
    F: Float + 'a,
    D: DistanceData<F> + ?Sized + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
    S: KnnSearch<F, D::Query<'a>>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances = compute_core_distances_tree(tree, data, min_points);

    let mut builder = ClusterBuilder::new(n);
    let mut primary = CandidateHeap::<F>::new();
    let mut neighbor_heaps: Vec<CandidateHeap<F>> = vec![CandidateHeap::new(); n];
    let mut searchers: Vec<Option<S::Searcher<'a>>> = (0..n).map(|_| None).collect();
    let mut node_cluster = vec![u32::MAX; n];

    let mut query = data.query();

    // initial pass: find the min_points nearest neighbors of each points
    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue; // duplicate, merged already
        }
        let mut searcher = tree.priority_searcher();
        query.set_index(a);
        initialize_neighbors(
            &query,
            &mut searcher,
            &mut builder,
            a,
            &mut neighbor_heaps[a],
            &core_distances,
        );
        if let Some(top) = neighbor_heaps[a].peek() {
            primary.push(DistPair::new(top.distance, a));
            searchers[a] = Some(searcher);
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = primary.pop() else {
            break;
        };
        let a = top.index;
        let nn = &mut neighbor_heaps[a];
        // Peek neighbor's best, handle same-cluster inline (like Java)
        while let Some(best) = nn.peek() {
            let b = best.index;
            if builder.find(a) != builder.find(b) {
                break; // different cluster, proceed to merge
            }
            nn.pop(); // discard same-cluster entry
        }
        // If neighbor heap has a valid candidate, try merge
        if let Some(best) = nn.peek().filter(|best| best.distance <= top.distance) {
            nn.pop();
            let b = best.index;
            builder.merge_points(a, b, best.distance);
            if builder.merge_count() == n - 1 {
                break;
            }
        }

        // Refill when the heap is empty or the best candidate exceeds the
        // search lower bound (i.e. unseen points could be closer).
        if let Some(ref mut searcher) = searchers[a] {
            let lb = searcher.all_lower_bound();
            if nn.peek().is_none_or(|next| next.distance > lb) {
                query.set_index(a);
                refill_neighbors(
                    &query,
                    searcher,
                    &mut builder,
                    a,
                    nn,
                    &core_distances,
                    &mut node_cluster,
                );
            }
        }

        if let Some(next) = nn.peek() {
            primary.push(DistPair::new(next.distance, a));
        } else {
            searchers[a] = None;
        }
    }

    HdbscanHierarchy::new(builder.into_history(), core_distances)
}

fn initialize_neighbors<F: Float, Q, S>(
    query: &Q, searcher: &mut S, builder: &mut ClusterBuilder<F>, query_index: usize,
    heap: &mut CandidateHeap<F>, core_distances: &[F],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let cd = core_distances[query_index];
    let mut threshold = F::infinity();
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next(query) else {
            break;
        };
        let b = cand.index;
        if b == query_index {
            continue; // skip self
        }
        if cand.distance == F::zero() {
            // Merge exact duplicates immediately at core distance
            let _ = builder.merge_points(query_index, b, cd);
            continue;
        }
        let dist = cd.max(core_distances[b]).max(cand.distance); // mutual reachability
        heap.push(DistPair::new(dist, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

fn refill_neighbors<F: Float, Q, S>(
    query: &Q, searcher: &mut S, builder: &mut ClusterBuilder<F>, query_index: usize,
    heap: &mut CandidateHeap<F>, core_distances: &[F], node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let query_component = builder.find(query_index);
    let cd = core_distances[query_index];
    // Purge stale same-cluster entries so threshold is not artificially low
    purge_same_cluster_component(heap, builder, query_component);
    let mut threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(
            query,
            &mut SameClusterFilter { builder, query_component, node_cluster },
        ) else {
            break;
        };
        let b = cand.index;
        let dist = cd.max(core_distances[b]).max(cand.distance); // mutual reachability
        heap.push(DistPair::new(dist, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

/// Purge same-cluster entries from the top of a min-heap.
fn purge_same_cluster_component<F: Float>(
    heap: &mut CandidateHeap<F>, builder: &mut ClusterBuilder<F>, query_component: usize,
) {
    while heap.peek().is_some_and(|n| builder.find(n.index) == query_component) {
        heap.pop();
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::heap_of_searchers_hdbscan;
    use crate::TableWithDistance;
    use crate::cluster::hdbscan::hdbscan_prim;
    use crate::distance::Euclidean;
    use crate::vptree::VPTree;

    #[test]
    fn heap_of_searchers_hdbscan_matches_prim_mst() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 1.2],
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(17);
        let tree = VPTree::<f64>::new(&data, 3, &mut rng);

        let expected = hdbscan_prim(&data, 2);
        let got = heap_of_searchers_hdbscan(&tree, &data, 2);
        assert_eq!(got, expected);
    }

    #[test]
    fn heap_of_searchers_hdbscan_matches_prim_with_duplicates() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 1.2],
            vec![1.0, 1.2], // exact duplicate
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(17);
        let tree = VPTree::<f64>::new(&data, 3, &mut rng);

        let expected = hdbscan_prim(&data, 2);
        let got = heap_of_searchers_hdbscan(&tree, &data, 2);
        let expected_weight: f64 = expected.merges.iter().map(|m| m.distance).sum();
        let got_weight: f64 = got.merges.iter().map(|m| m.distance).sum();
        assert!(
            (expected_weight - got_weight).abs() < 1e-10,
            "MST weights differ: prim={expected_weight}, heap={got_weight}"
        );
    }
}
