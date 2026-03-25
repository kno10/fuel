use std::collections::BinaryHeap;

use crate::api::{DistanceData, DistanceSearch, PrioritySearcher, PrioritySearcherFactory};
use crate::cluster::hierarchical::common::MergeHistory;
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{DistPair, Float, IndexQuery};

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
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(slack > 0, "slack must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut primary = BinaryHeap::<DistPair<F>>::new();
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
            // breaking ties by smaller index). With reversed `DistPair` ordering,
            // a better candidate is > worst.
            let worst = worst_candidate(buffer);
            if candidate > worst {
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
    // `DistPair` ordering is reversed so that smaller distances are "greater".
    // Here we want the worst one (largest distance, then largest index), which is
    // the minimal `DistPair` under reversed ordering.
    *buffer.iter().min().expect("buffer should not be empty")
}

/// Replace the worst entry in the buffer with `item`.
fn replace_worst<F: Float>(buffer: &mut [DistPair<F>], item: DistPair<F>) {
    if let Some(idx) = buffer.iter().enumerate().min_by_key(|(_, v)| *v).map(|(i, _)| i) { 
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
    use crate::TableWithDistance;
    use crate::data::CondensedDistanceMatrix;
    use crate::distance::EuclideanDistance;
    use crate::vptree::VPTree;

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
    fn buffered_matches_slink_on_unique_1d_distances() {
        let points = vec![vec![0.0], vec![1.1], vec![3.7], vec![10.2], vec![20.5]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(11);
        let tree = VPTree::new(&data, 3, &mut rng);

        let vec = condensed_abs_1d(&points);
        let cm = CondensedDistanceMatrix::new(&vec, points.len());
        let expected = crate::cluster::hierarchical::slink(&cm);
        let got = buffered_search_single_link(&tree, &data, 2);
        assert_eq!(got, expected);
    }
}
