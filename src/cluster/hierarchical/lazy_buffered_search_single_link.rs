use crate::api::{DistanceData, DistanceSearch, PrioritySearcher, PrioritySearcherFactory};
use crate::cluster::hierarchical::common::{BufferedNeighbors, MergeHistory};
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{CandidateHeap, DistPair, Float, IndexQuery};

/// Lazy Buffered-Search Single-Link with VP-tree priority search.
///
/// This variant keeps an unbounded buffer per point and uses a `SameClusterFilter`
/// with a witness cache for skip_node pruning.  `slack` controls how many extra
/// candidates are explored beyond the current lower-bound threshold before
/// stopping each refill phase.
#[must_use]
pub fn lazy_buffered_search_single_link<'a, S, D, F>(
    tree: &'a S, data: &'a D, slack: usize,
) -> MergeHistory<F>
where
    F: Float + 'a,
    D: DistanceData<F> + ?Sized + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut primary = CandidateHeap::<F>::new();
    let mut buffers: Vec<BufferedNeighbors<F>> =
        (0..n).map(|_| BufferedNeighbors::<F>::new()).collect();
    let mut node_cluster = vec![u32::MAX; n];

    // create one searcher and reuse it for all refill operations
    let mut searcher = tree.priority_searcher();

    let mut query = data.query();

    // initial fill for each point
    for (a, buf) in buffers.iter_mut().enumerate().take(n) {
        if builder.cluster_size_of_point(a) > 1 {
            continue; // duplicate, merged already
        }
        query.set_index(a);
        refill_neighbors(
            &query,
            &mut builder,
            a,
            F::zero(),
            slack,
            buf.reset(),
            &mut searcher,
            &mut node_cluster,
        );
        if let Some(top) = buf.peek() {
            primary.push(DistPair::new(top.distance, a));
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = primary.pop() else {
            break;
        };
        let a = top.index;
        let buf = &mut buffers[a];

        // Purge same-cluster entries from the top of the buffer.
        purge_same_cluster(buf, &mut builder, a);

        // Only merge when the best candidate's distance is consistent with
        // the queued distance.  If a chain of merges made the queued distance
        // stale the merge is skipped, but we still fall through to the refill
        // check so the primary is re-queued with the best known distance after
        // the search is advanced.
        if let Some(best) = buf.peek().filter(|b| b.distance <= top.distance) {
            buf.pop();
            let b = best.index;
            if builder.merge_points(a, b, best.distance).is_some() && builder.merge_count() == n - 1
            {
                break;
            }
            // Purge items that became same-cluster due to the merge so that
            // needs_refill sees the true nearest non-same-cluster distance.
            purge_same_cluster(buf, &mut builder, a);
        }

        let needs_refill = buf.peek().is_none()
            || buf.peek().map_or(F::infinity(), |n| n.distance) > buf.threshold;
        if needs_refill {
            query.set_index(a);
            refill_neighbors(
                &query,
                &mut builder,
                a,
                buf.threshold,
                slack,
                buf,
                &mut searcher,
                &mut node_cluster,
            );
        }

        if let Some(next) = buf.peek() {
            primary.push(DistPair::new(next.distance, a));
        }
    }

    builder.into_history()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refill_neighbors<F: Float, Q, S>(
    query: &Q, builder: &mut ClusterBuilder<F>, query_index: usize, skip: F, slack: usize,
    buffer: &mut BufferedNeighbors<F>, searcher: &mut S, node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let mut threshold = buffer.peek().map_or(F::infinity(), |n| n.distance);
    let mut remaining = slack as isize;
    let query_component = builder.find(query_index);
    searcher.reset_with_limits(F::infinity(), skip.max(F::zero()));

    let mut filter: SameClusterFilter<'_, F> =
        SameClusterFilter { builder, query_component, node_cluster };

    while searcher.all_lower_bound() < threshold && remaining > 0 {
        let Some(cand) = searcher.next_with_filter(query, &mut filter) else {
            break;
        };
        let b = cand.index;
        let d = cand.distance;
        if d < skip {
            continue;
        }
        buffer.push(DistPair::new(d, b));
        threshold = buffer.peek().map_or(F::infinity(), |n| n.distance);
        if searcher.all_lower_bound() >= threshold {
            remaining -= 1;
        }
    }

    buffer.threshold = searcher.all_lower_bound();
}

pub(crate) fn purge_same_cluster<F: Float>(
    buf: &mut BufferedNeighbors<F>, builder: &mut ClusterBuilder<F>, a: usize,
) {
    if !buf.is_empty() {
        let ca = builder.find(a);
        while let Some(candidate) = buf.peek()
            && builder.find(candidate.index) == ca
        {
            buf.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;
    use crate::{CondensedDistanceMatrix, TableWithDistance};

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
    fn lazy_buffered_matches_slink_on_unique_1d_distances() {
        let points = vec![vec![0.0], vec![1.1], vec![3.7], vec![10.2], vec![20.5]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(11);
        let tree = VPTree::new(&data, 3, &mut rng);

        let vec = condensed_abs_1d(&points);
        let cm = CondensedDistanceMatrix::new(&vec, points.len());
        let expected = crate::cluster::hierarchical::slink(&cm);
        let got = lazy_buffered_search_single_link(&tree, &data, 2);
        assert_eq!(got, expected);
    }
}
