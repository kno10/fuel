use std::collections::BinaryHeap;

use num_traits::Float;

use crate::api::{DistanceData, DistanceSearch};
use crate::cluster::hierarchical::common::MergeHistory;
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::DistPair;
use crate::vptree::{PrioritySearcher, VPTree};

/// Restarting-Search Single-Link (RSSL) with VP-tree priority search.
///
/// This algorithm is similar to a buffered search with `slack = 1`.
/// DO NOT DELEGATE TO buffered_search_single_link. This version MUST store only a single next nearest neighbor for each point.
#[must_use]
pub fn restarting_search_single_link<D: DistanceData<F>, F: Float>(
    tree: &VPTree<F>,
    data: &D,
) -> MergeHistory<F> {
    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut primary = BinaryHeap::<DistPair<F>>::new();
    let mut buffers: Vec<DistPair<F>> = vec![DistPair::undefined(); n];
    let mut node_cluster = vec![u32::MAX; n];

    // create one searcher and reuse it for all refill operations
    let mut searcher = tree.priority_searcher();

    // initial fill for each point
    for (a, buf) in buffers.iter_mut().enumerate().take(n) {
        if builder.cluster_size_of_point(a) > 1 {
            continue; // duplicate, merged already
        }
        refill_neighbors(
            &data.search_by_index(a),
            &mut builder,
            a,
            F::zero(),
            buf,
            &mut searcher,
            &mut node_cluster,
        );
        if !buf.is_sentinel() {
            primary.push(DistPair::new(buf.distance, a));
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = primary.pop() else {
            break;
        };
        let a = top.index;
        let buf = &mut buffers[a];

        if buf.is_sentinel() {
            continue;
        }
        let best = std::mem::replace(buf, DistPair::undefined());

        let best_dist = best.distance;
        let b = best.index;
        if builder.merge_points(a, b, best_dist).is_some() {
            if builder.merge_count() == n - 1 {
                break;
            }
        }

        refill_neighbors(
            &data.search_by_index(a),
            &mut builder,
            a,
            best_dist,
            buf,
            &mut searcher,
            &mut node_cluster,
        );

        if !buf.is_sentinel() {
            primary.push(DistPair::new(buf.distance, a));
        }
    }

    builder.into_history()
}

pub(crate) fn refill_neighbors<D: DistanceSearch<F>, F: Float>(
    data: &D,
    builder: &mut ClusterBuilder<F>,
    query_index: usize,
    skip: F,
    buffer: &mut DistPair<F>,
    searcher: &mut PrioritySearcher<F>,
    node_cluster: &mut [u32],
) {
    searcher.reset_with_limits(F::infinity(), skip.max(F::zero()));

    let mut threshold = F::infinity();
    let query_component = builder.find(query_index);
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
        if d < threshold {
            *buffer = DistPair::new(d, b);
            threshold = d;
            searcher.decrease_cutoff(d);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableWithDistance;
    use crate::cluster::hierarchical::buffered_search_single_link;
    use crate::distance::EuclideanDistance;
    use num_traits::ToPrimitive;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Ensure that restarting search produces the same merge history as a
    /// buffered search with slack=1.  This also serves as a regression test
    /// for the bug that caused RSSL to revisit neighbours and run slowly.
    #[test]
    fn restarting_equals_buffered_random() {
        // generate a few random 2‑D points and compare results
        let mut rng = StdRng::seed_from_u64(42);
        let points: Vec<Vec<f64>> = (0..20)
            .map(|_| vec![rng.r#gen::<f64>(), rng.r#gen::<f64>()])
            .collect();

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree = VPTree::<f64>::new(&data, 3, &mut rng);

        let hist_r = restarting_search_single_link(&tree, &data);
        let hist_b = buffered_search_single_link(&tree, &data, 1);
        assert_eq!(hist_r.len(), hist_b.len());
        // sort both histories by (idx1, idx2) to allow differences in merge order
        let mut r_sorted = hist_r.clone();
        let mut b_sorted = hist_b.clone();
        r_sorted.sort_by(|a, c| (a.idx1, a.idx2).cmp(&(c.idx1, c.idx2)));
        b_sorted.sort_by(|a, c| (a.idx1, a.idx2).cmp(&(c.idx1, c.idx2)));
        for (r, b) in r_sorted.iter().zip(b_sorted.iter()) {
            assert_eq!(r.idx1, b.idx1);
            assert_eq!(r.idx2, b.idx2);
            assert_eq!(r.size, b.size);
            // allow tiny floating-point discrepancies
            let diff: f64 = (r.distance.to_f64().unwrap() - b.distance.to_f64().unwrap()).abs();
            assert!(diff < 1e-6, "distance mismatch {r:?} vs {b:?}");
        }
    }
}
