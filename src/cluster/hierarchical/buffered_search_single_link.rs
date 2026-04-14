use crate::api::{DistanceData, DistanceSearch, PrioritySearcher, PrioritySearcherFactory};
use crate::cluster::hierarchical::MergeHistory;
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{CandidateHeap, DistPair, Float, IndexQuery};

/// Buffered-Search Single-Link (BSSL) with VP-tree priority search.
///
/// Each point maintains a bounded buffer of at most `slack` candidate
/// neighbors.  When the buffer runs dry it is refilled by continuing the
/// priority search from where the previous distance threshold left off.
/// Unlike `lazy_buffered_search_single_link`, this variant caps memory per
/// point and does not track "seen" points; instead it relies on the
/// `SameClusterFilter` with a witness cache for skip_node pruning.
#[must_use]
pub fn buffered_search_single_link<'a, S, D, F>(
    tree: &'a S, data: &'a D, slack: usize,
) -> MergeHistory<F>
where
    F: Float + 'a,
    D: DistanceData<F> + ?Sized + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    assert!(slack > 0, "slack must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut primary = CandidateHeap::<F>::new();
    // Per-point bounded buffers (sorted ascending by distance, best at end).
    let mut buffers: Vec<Vec<DistPair<F>>> = (0..n).map(|_| Vec::with_capacity(slack)).collect();
    let mut skip: Vec<F> = vec![F::zero(); n];
    let mut node_cluster = vec![u32::MAX; n];

    let mut searcher = tree.priority_searcher();

    let mut query = data.query();

    // initial fill
    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        query.set_index(a);
        refill_buffer(
            &query,
            &mut builder,
            a,
            skip[a],
            slack,
            &mut buffers[a],
            &mut searcher,
            &mut node_cluster,
        );
        if let Some(best) = buffers[a].last() {
            primary.push(DistPair::new(best.distance, a));
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = primary.pop() else {
            break;
        };
        let a = top.index;
        let buf = &mut buffers[a];

        // Purge same-cluster entries from the buffer.
        purge_same_cluster(buf, &mut builder, a);

        if buf.is_empty() {
            // Buffer emptied by purge; attempt refill.
            query.set_index(a);
            refill_buffer(
                &query,
                &mut builder,
                a,
                skip[a],
                slack,
                buf,
                &mut searcher,
                &mut node_cluster,
            );
            if buf.is_empty() {
                continue;
            }
        }

        let best = *buf.last().unwrap(); // best = smallest distance (end of desc-sorted vec)
        if best.distance > top.distance {
            // Purge changed the best; re-insert with corrected priority.
            primary.push(DistPair::new(best.distance, a));
            continue;
        }
        buf.pop();

        let best_dist = best.distance;
        let b = best.index;
        if builder.merge_points(a, b, best_dist).is_some() && builder.merge_count() == n - 1 {
            break;
        }
        skip[a] = best_dist;

        // Refill if buffer ran dry.
        if buf.is_empty() {
            query.set_index(a);
            refill_buffer(
                &query,
                &mut builder,
                a,
                skip[a],
                slack,
                buf,
                &mut searcher,
                &mut node_cluster,
            );
        }

        if let Some(next) = buf.last() {
            primary.push(DistPair::new(next.distance, a));
        }
    }

    builder.into_history()
}

/// Fill `buffer` with up to `slack` nearest not-same-cluster neighbors,
/// starting the search from distance `skip` onwards.
///
/// The buffer is cleared and refilled from scratch each time.  Entries are
/// stored in **descending** distance order so that `last()` gives the best
/// (closest) and `first()` gives the worst (farthest) for easy eviction.
#[allow(clippy::too_many_arguments)]
pub(crate) fn refill_buffer<F: Float, Q, S>(
    query: &Q, builder: &mut ClusterBuilder<F>, query_index: usize, skip: F, slack: usize,
    buffer: &mut Vec<DistPair<F>>, searcher: &mut S, node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    buffer.clear();
    searcher.reset_with_limits(F::infinity(), skip.max(F::zero()));

    let query_component = builder.find(query_index);

    let mut threshold = F::infinity();
    let mut filter: SameClusterFilter<'_, F> =
        SameClusterFilter { builder, query_component, node_cluster };

    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(query, &mut filter) else {
            break;
        };
        let b = cand.index;
        let d = cand.distance;

        if d < skip {
            continue;
        }

        let candidate = DistPair::new(d, b);

        if buffer.len() < slack {
            buffer.push(candidate);
            if buffer.len() == slack {
                // Buffer full -- tighten the search radius.
                threshold = worst_candidate(buffer).distance;
                searcher.decrease_cutoff(threshold);
            }
        } else {
            // Replace the worst candidate if this one is better (smaller distance,
            // breaking ties by smaller index). With natural `DistPair` ordering,
            // a better candidate is < worst.
            let worst = worst_candidate(buffer);
            if candidate < worst {
                replace_worst(buffer, candidate);
                threshold = worst_candidate(buffer).distance;
                searcher.decrease_cutoff(threshold);
            }
        }
    }

    // Sort descending so best (smallest distance) is at the end for pop().
    buffer.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
}

/// Get the worst candidate in the buffer according to the `DistPair` ordering.
fn worst_candidate<F: Float>(buffer: &[DistPair<F>]) -> DistPair<F> {
    // We want the worst one (largest distance, then largest index), which is
    // the maximal `DistPair` under natural ordering.
    *buffer.iter().max().expect("buffer should not be empty")
}

/// Replace the worst entry in the buffer with `item`.
fn replace_worst<F: Float>(buffer: &mut [DistPair<F>], item: DistPair<F>) {
    if let Some(idx) = buffer.iter().enumerate().max_by_key(|(_, v)| *v).map(|(i, _)| i) {
        buffer[idx] = item;
    }
}

/// Remove buffer entries whose point is now in the same cluster as `a`.
fn purge_same_cluster<F: Float>(
    buffer: &mut Vec<DistPair<F>>, builder: &mut ClusterBuilder<F>, a: usize,
) {
    let ca = builder.find(a);
    buffer.retain(|n| builder.find(n.index) != ca);
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
    fn buffered_search_single_link_regression() {
        test_clustering_table(
            "BufferedSearchSingleLink",
            "single",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let mut rng = StdRng::seed_from_u64(42);
                let tree = VPTree::new(access, 3, &mut rng);
                let history = buffered_search_single_link(&tree, access, 1);
                cut_dendrogram_by_number_of_clusters(&history, min_clusters)
            },
        );
    }
}
