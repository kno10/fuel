use crate::api::{DistanceData, DistanceSearch, PrioritySearcher, PrioritySearcherFactory};
use crate::cluster::hierarchical::common::MergeHistory;
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{CandidateHeap, DistPair, Float, IndexQuery};

/// Heap-of-Searchers Single-Link (HSSL) with priority-search acceleration.
#[must_use]
pub fn heap_of_searchers_single_link<'a, S, D, F>(tree: &'a S, data: &'a D) -> MergeHistory<F>
where
    F: Float + 'a,
    D: DistanceData<F> + ?Sized + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut primary = CandidateHeap::<F>::new();
    let mut neighbor_heaps: Vec<CandidateHeap<F>> = vec![CandidateHeap::new(); n];
    let mut searchers: Vec<Option<S::Searcher<'a>>> = (0..n).map(|_| None).collect();
    let mut node_cluster = vec![u32::MAX; n];

    let mut query = data.query();

    // initial pass: find the 1-nearest neighbor for each point
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
            &mut node_cluster,
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

        // Purge stale same-cluster entries from the top of the heap.
        purge_same_cluster(nn, &mut builder, a);

        // Only merge when the best candidate's distance is consistent with
        // the distance at which `a` was queued.  If a recent merge made the
        // queued distance stale (heap min is now farther), defer via re-queue
        // but still fall through to the refill below.
        if let Some(best) = nn.peek().filter(|b| b.distance <= top.distance) {
            nn.pop();
            let b = best.index;
            if builder.merge_points(a, b, best.distance).is_some() && builder.merge_count() == n - 1
            {
                break;
            }
        }

        // Refill when the heap is empty or the best candidate exceeds the
        // VP-tree lower bound.  Always reached — even when no merge happened
        // due to a stale top distance — so the primary is always re-queued
        // with the best known non-same-cluster distance after the refill.
        if let Some(searcher) = searchers[a].as_mut() {
            let lb = searcher.all_lower_bound();
            if nn.peek().is_none_or(|next| next.distance > lb) {
                query.set_index(a);
                refill_neighbors(&query, searcher, &mut builder, a, nn, &mut node_cluster);
            }
        }

        if let Some(next) = nn.peek() {
            primary.push(DistPair::new(next.distance, a));
        } else {
            searchers[a] = None;
        }
    }

    builder.into_history()
}

// `initialize_neighbors` performs an unfiltered priority search, pushing
// the nearest neighbors into `heap`.  Exact duplicates are merged eagerly
// since they represent zero-distance points and are handled specially by
// the algorithm.
fn initialize_neighbors<F: Float, Q, S>(
    query: &Q, searcher: &mut S, builder: &mut ClusterBuilder<F>, a: usize,
    heap: &mut CandidateHeap<F>, node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    let mut threshold = F::infinity();
    while searcher.all_lower_bound() < threshold {
        let (b, d) = {
            let query_component = builder.find(a);
            let mut filter: SameClusterFilter<'_, F> =
                SameClusterFilter { builder, query_component, node_cluster };
            let Some(cand) = searcher.next_with_filter(query, &mut filter) else {
                break;
            };
            (cand.index, cand.distance)
        };
        if d == F::zero() {
            // merge any exact duplicates immediately
            let _ = builder.merge_points(a, b, F::zero());
            continue;
        }
        heap.push(DistPair::new(d, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

fn refill_neighbors<F: Float, Q, S>(
    query: &Q, searcher: &mut S, builder: &mut ClusterBuilder<F>, query_index: usize,
    heap: &mut CandidateHeap<F>, node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    // Purge stale same-cluster entries so the threshold is not artificially
    // capped by a neighbour that merged into the query's cluster since it
    // was first discovered.
    purge_same_cluster(heap, builder, query_index);
    let query_component = builder.find(query_index);
    let mut threshold = heap.peek().map_or(F::infinity(), |n| n.distance);

    let mut filter: SameClusterFilter<'_, F> =
        SameClusterFilter { builder, query_component, node_cluster };

    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(query, &mut filter) else {
            break;
        };
        let b = cand.index;
        heap.push(DistPair::new(cand.distance, b));
        threshold = heap.peek().map_or(F::infinity(), |n| n.distance);
    }
}

pub(crate) fn purge_same_cluster<F: Float>(
    heap: &mut CandidateHeap<F>, builder: &mut ClusterBuilder<F>, a: usize,
) {
    let ca = builder.find(a);
    while heap.peek().is_some_and(|n| builder.find(n.index) == ca) {
        heap.pop();
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::cluster::hierarchical::restarting_search_single_link;
    use crate::distance::{DistanceFunction, Euclidean};
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
    fn hssl_matches_slink_on_unique_1d_distances() {
        let points = vec![vec![0.0], vec![1.1], vec![3.7], vec![10.2], vec![20.5]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(17);
        let tree = VPTree::new(&data, 3, &mut rng);

        let vec = condensed_abs_1d(&points);
        let cm = CondensedDistanceMatrix::new_from_condensed(vec, points.len());
        let expected = crate::cluster::hierarchical::slink(&cm);
        let got = heap_of_searchers_single_link(&tree, &data);
        assert_eq!(got, expected);
    }

    #[test]
    fn hssl_distance_count_not_worse_than_rssl() {
        use std::cell::Cell;

        struct CountingDist<'a> {
            counter: &'a Cell<usize>,
        }

        impl DistanceFunction<Vec<f64>, f64> for CountingDist<'_> {
            fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
                self.counter.set(self.counter.get() + 1);
                Euclidean.distance(a, b)
            }
        }

        impl DistanceFunction<[f64], f64> for CountingDist<'_> {
            fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
                self.counter.set(self.counter.get() + 1);
                Euclidean.distance(a, b)
            }
        }

        use rand::RngExt;

        let mut rng = StdRng::seed_from_u64(42);
        let points: Vec<Vec<f64>> =
            (0..30).map(|_| vec![rng.random::<f64>(), rng.random::<f64>()]).collect();
        let counter1 = Cell::new(0);
        let data1: TableWithDistance<f64, Vec<f64>, CountingDist, f64> =
            TableWithDistance::with_distance(&points, CountingDist { counter: &counter1 });
        let mut rng = StdRng::seed_from_u64(42);
        let tree1: VPTree<f64> = VPTree::new(&data1, 3, &mut rng);
        let _ = heap_of_searchers_single_link(&tree1, &data1);
        let dist_hssl = counter1.get();

        let counter2 = Cell::new(0);
        let data2: TableWithDistance<f64, Vec<f64>, CountingDist, f64> =
            TableWithDistance::with_distance(&points, CountingDist { counter: &counter2 });
        let mut rng = StdRng::seed_from_u64(42);
        let tree2: VPTree<f64> = VPTree::new(&data2, 3, &mut rng);
        let _ = restarting_search_single_link(&tree2, &data2);
        let dist_rssl = counter2.get();

        assert!(
            dist_hssl <= dist_rssl,
            "HSSL used {dist_hssl} distances but RSSL used {dist_rssl}"
        );
    }
}
