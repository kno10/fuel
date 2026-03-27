use super::hdbscan_common::{HdbscanHierarchy, compute_core_distances_tree};
use crate::api::DistanceData;
use crate::cluster::hierarchical::common::BufferedNeighbors;
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{
    CandidateHeap, DistPair, DistanceSearch, Float, IndexQuery, KnnSearch, PrioritySearcher,
    PrioritySearcherFactory,
};
/// Lazy buffered-search HDBSCAN MST.
///
/// This variant keeps an unbounded buffer per point and uses a `SameClusterFilter`
/// with a witness cache for skip_node pruning.  The slack parameter controls how
/// many extra priority-search expansions are allowed beyond the current
/// lower-bound threshold.
#[must_use]
pub fn lazy_buffered_search_hdbscan<'a, S, D, F>(
    tree: &'a S, data: &'a D, min_points: usize, slack: usize,
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
    if n == 1 {
        return HdbscanHierarchy::new(Vec::new(), core_distances);
    }

    let mut builder = ClusterBuilder::new(n);
    let mut heap = CandidateHeap::<F>::new();
    let mut neighbor_buffers: Vec<BufferedNeighbors<F>> =
        (0..n).map(|_| BufferedNeighbors::new()).collect();
    let mut thresholds = vec![F::infinity(); n];
    let mut node_cluster = vec![u32::MAX; n];
    let mut searcher = S::priority_searcher(tree);

    let mut query = data.query();

    // Fill initial heaps
    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        query.set_index(a);
        thresholds[a] = refill_neighbors(
            &mut builder,
            a,
            &query,
            F::zero(),
            slack,
            &mut neighbor_buffers[a],
            &mut searcher,
            &core_distances,
            &mut node_cluster,
        );
        if let Some(top) = neighbor_buffers[a].peek() {
            heap.push(DistPair::new(top.distance, a));
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(entry) = heap.pop() else {
            break;
        };
        let a = entry.index;
        let buffer = &mut neighbor_buffers[a];
        while let Some(candidate) = buffer.peek() {
            if builder.same_set(a, candidate.index) {
                buffer.pop();
            } else {
                break;
            }
        }
        // Only merge when the best candidate's distance is consistent with
        // the queued distance.  If a chain of merges made the queued distance
        // stale the merge is skipped, but we still fall through to the refill
        // check so the heap is re-queued with the best known distance after
        // the search is advanced.
        if let Some(best) = buffer.peek().filter(|b| b.distance <= entry.distance) {
            buffer.pop();
            let b = best.index;
            let best_dist = best.distance;
            if builder.merge_points(a, b, best_dist).is_some() && builder.merge_count() == n - 1 {
                break;
            }
            // Purge items that became same-cluster due to the merge so that
            // needs_refill sees the true nearest non-same-cluster distance.
            while let Some(candidate) = buffer.peek() {
                if builder.same_set(a, candidate.index) {
                    buffer.pop();
                } else {
                    break;
                }
            }
        }

        let needs_refill = buffer.peek().is_none()
            || buffer.peek().map(|n| n.distance).unwrap_or(F::infinity()) > thresholds[a];
        if needs_refill {
            query.set_index(a);
            thresholds[a] = refill_neighbors(
                &mut builder,
                a,
                &query,
                thresholds[a],
                slack,
                buffer,
                &mut searcher,
                &core_distances,
                &mut node_cluster,
            );
        }

        if let Some(next) = buffer.peek() {
            heap.push(DistPair::new(next.distance, a));
        }
    }

    HdbscanHierarchy::new(builder.into_history(), core_distances)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refill_neighbors<F: Float, Q, S>(
    builder: &mut ClusterBuilder<F>, query_index: usize, query: &Q, skip: F, slack: usize,
    buffer: &mut BufferedNeighbors<F>, searcher: &mut S, core_distances: &[F],
    node_cluster: &mut [u32],
) -> F
where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    searcher.reset_with_limits(F::infinity(), skip.max(F::zero()));

    let cd = core_distances[query_index];
    let mut threshold = buffer.peek().map_or(F::infinity(), |n| n.distance);
    let mut remaining = slack as isize;
    let query_component = builder.find(query_index);
    loop {
        let lower_bound = searcher.all_lower_bound();
        if lower_bound >= threshold && remaining <= 0 {
            break;
        }
        let Some(cand) = searcher.next_with_filter(
            query,
            &mut SameClusterFilter { builder, query_component, node_cluster },
        ) else {
            break;
        };
        let b = cand.index;
        let d = cand.distance;
        if d < skip {
            continue;
        }
        let rd = cd.max(core_distances[b]).max(d); // mutual reachability
        buffer.push(DistPair::new(rd, b));
        threshold = buffer.peek().map_or(F::infinity(), |n| n.distance);
        if lower_bound >= threshold {
            remaining -= 1;
        }
    }

    buffer.threshold = searcher.all_lower_bound();
    buffer.threshold
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::super::hdbscan_prim;
    use super::lazy_buffered_search_hdbscan;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::vptree::VPTree;

    #[test]
    fn buffered_search_matches_linear_mst() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![1.0, 1.2],
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(11);
        let tree = VPTree::<f64>::new(&data, 3, &mut rng);

        let expected = hdbscan_prim(&data, 2);
        let got = lazy_buffered_search_hdbscan(&tree, &data, 2, 1);
        assert_eq!(got, expected);
    }
}
