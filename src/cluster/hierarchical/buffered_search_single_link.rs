use std::collections::BinaryHeap;

use num_traits::Float;

use crate::api::{DistanceData, DistanceSearch};
use crate::cluster::hierarchical::common::MergeHistory;
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::DistPair;
use crate::vptree::{PrioritySearcher, VPTree};

/// Buffered-Search Single-Link (BSSL) with VP-tree priority search.
///
/// Each point maintains a bounded buffer of at most `slack` candidate
/// neighbors.  When the buffer runs dry it is refilled by continuing the
/// priority search from where the previous distance threshold left off.
/// Unlike `lazy_buffered_search_single_link`, this variant caps memory per
/// point and does not track "seen" points; instead it relies on the
/// `SameClusterFilter` with a witness cache for skip_node pruning.
#[must_use]
pub fn buffered_search_single_link<D: DistanceData<F>, F: Float>(
    tree: &VPTree<F>,
    data: &D,
    slack: usize,
) -> MergeHistory<F> {
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

    // initial fill
    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        refill_buffer(
            &data.search_by_index(a),
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
            refill_buffer(
                &data.search_by_index(a),
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
        if builder.merge_points(a, b, best_dist).is_some() {
            if builder.merge_count() == n - 1 {
                break;
            }
        }
        skip[a] = best_dist;

        // Refill if buffer ran dry.
        if buf.is_empty() {
            refill_buffer(
                &data.search_by_index(a),
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
fn refill_buffer<D: DistanceSearch<F>, F: Float>(
    data: &D,
    builder: &mut ClusterBuilder<F>,
    query_index: usize,
    skip: F,
    slack: usize,
    buffer: &mut Vec<DistPair<F>>,
    searcher: &mut PrioritySearcher<F>,
    node_cluster: &mut [u32],
) {
    buffer.clear();
    searcher.reset_with_limits(F::infinity(), skip.max(F::zero()));

    let query_component = builder.find(query_index);
    let mut threshold = F::infinity();

    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(
            data,
            &mut SameClusterFilter {
                builder,
                query_component,
                node_cluster,
            },
        ) else {
            break;
        };
        let b = cand.index;
        let d = cand.distance;

        if buffer.len() < slack {
            buffer.push(DistPair::new(d, b));
            if buffer.len() == slack {
                // Buffer full -- tighten the search radius.
                threshold = worst_distance(buffer);
                searcher.decrease_cutoff(threshold);
            }
        } else if d < threshold {
            // Replace worst (first element = largest distance).
            replace_worst(buffer, DistPair::new(d, b));
            threshold = worst_distance(buffer);
            searcher.decrease_cutoff(threshold);
        }
    }

    // Sort descending so best (smallest distance) is at the end for pop().
    buffer.sort_by(|a, b| {
        b.distance
            .partial_cmp(&a.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Distance of the worst (largest) entry in the buffer.
fn worst_distance<F: Float>(buffer: &[DistPair<F>]) -> F {
    buffer
        .iter()
        .map(|n| n.distance)
        .fold(F::neg_infinity(), |a, b| if a > b { a } else { b })
}

/// Replace the worst (largest distance) entry with `item`.
fn replace_worst<F: Float>(buffer: &mut [DistPair<F>], item: DistPair<F>) {
    if let Some(idx) = buffer
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
    {
        buffer[idx] = item;
    }
}

/// Remove buffer entries whose point is now in the same cluster as `a`.
fn purge_same_cluster<F: Float>(
    buffer: &mut Vec<DistPair<F>>,
    builder: &mut ClusterBuilder<F>,
    a: usize,
) {
    let ca = builder.find(a);
    buffer.retain(|n| builder.find(n.index) != ca);
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::TableWithDistance;
    use crate::data::CondensedDistanceMatrix;
    use crate::distance::EuclideanDistance;

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
