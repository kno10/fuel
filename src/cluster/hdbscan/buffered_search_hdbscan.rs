use super::hdbscan_common::{HdbscanHierarchy, compute_core_distances_tree};
use crate::api::DistanceData;
use crate::cluster::hierarchical::MergeHistory;
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{
    CandidateHeap, DistPair, DistanceSearch, Float, IndexQuery, KnnSearch, PrioritySearcher,
    PrioritySearcherFactory,
};

/// Buffered-search HDBSCAN MST with bounded per-point buffers.
///
/// Each point maintains a buffer of at most `slack` candidate neighbors
/// (using mutual reachability distance).  When a buffer runs dry it is
/// refilled by restarting the priority search from the previous distance
/// threshold.  Unlike `lazy_buffered_search_hdbscan`, this variant caps
/// memory per point and relies on `SameClusterFilter` with a witness cache
/// for skip_node pruning.
pub fn buffered_search_hdbscan<'a, S, D, F>(
    tree: &'a S, data: &'a D, min_points: usize, slack: usize,
) -> Result<HdbscanHierarchy<F>, String>
where
    F: Float + 'a,
    D: DistanceData<F> + ?Sized + 'a,
    S: PrioritySearcherFactory<F, D::Query<'a>>,
    S: KnnSearch<F, D::Query<'a>>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");
    assert!(slack > 0, "slack must be positive");

    let core_distances = compute_core_distances_tree(tree, data, min_points)?;
    if n == 1 {
        return Ok(HdbscanHierarchy::new(MergeHistory::new(), core_distances));
    }

    let mut builder = ClusterBuilder::new(n);
    let mut primary = CandidateHeap::<F>::new();
    let mut buffers: Vec<Vec<DistPair<F>>> = (0..n).map(|_| Vec::with_capacity(slack)).collect();
    let mut node_cluster = vec![u32::MAX; n];
    let mut searcher = S::priority_searcher(tree);

    let mut query = data.query();

    // initial fill
    for (a, buf) in buffers.iter_mut().enumerate() {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        query.set_index(a);
        refill_buffer(
            &mut builder,
            a,
            &query,
            slack,
            buf,
            &mut searcher,
            &core_distances,
            &mut node_cluster,
        );
        if let Some(best) = buf.last() {
            primary.push(DistPair::new(best.distance, a));
        }
    }

    while builder.merge_count() < n - 1 {
        crate::poll_interrupted()?;
        let Some(top) = primary.pop() else {
            break;
        };
        let a = top.index;
        let buf = &mut buffers[a];

        purge_same_cluster(buf, &mut builder, a);

        if buf.is_empty() {
            query.set_index(a);
            refill_buffer(
                &mut builder,
                a,
                &query,
                slack,
                buf,
                &mut searcher,
                &core_distances,
                &mut node_cluster,
            );
            if buf.is_empty() {
                continue;
            }
        }

        let best = *buf.last().unwrap();
        if best.distance > top.distance {
            primary.push(DistPair::new(best.distance, a));
            continue;
        }
        buf.pop();

        let best_dist = best.distance;
        let b = best.index;
        if builder.merge_points(a, b, best_dist).is_some() && builder.merge_count() == n - 1 {
            break;
        }

        if buf.is_empty() {
            query.set_index(a);
            refill_buffer(
                &mut builder,
                a,
                &query,
                slack,
                buf,
                &mut searcher,
                &core_distances,
                &mut node_cluster,
            );
        }

        if let Some(next) = buf.last() {
            primary.push(DistPair::new(next.distance, a));
        }
    }

    Ok(HdbscanHierarchy::new(builder.into_history(), core_distances))
}

/// Fill `buffer` with up to `slack` nearest not-same-cluster neighbors
/// using mutual reachability distance.
///
/// The buffer is stored in descending distance order so that `last()` gives
/// the best (closest) and `first()` gives the worst (farthest).
#[allow(clippy::too_many_arguments)]
fn refill_buffer<F: Float, Q, S>(
    builder: &mut ClusterBuilder<F>, query_index: usize, query: &Q, slack: usize,
    buffer: &mut Vec<DistPair<F>>, searcher: &mut S, core_distances: &[F],
    node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    buffer.clear();
    searcher.reset();

    let cd = core_distances[query_index];
    let query_component = builder.find(query_index);
    let mut threshold = F::infinity();

    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(
            query,
            &mut SameClusterFilter { builder, query_component, node_cluster },
        ) else {
            break;
        };
        let b = cand.index;
        let rd = cd.max(core_distances[b]).max(cand.distance); // mutual reachability

        if buffer.len() < slack {
            buffer.push(DistPair::new(rd, b));
            if buffer.len() == slack {
                threshold = worst_distance(buffer);
                searcher.decrease_cutoff(threshold);
            }
        } else if rd < threshold {
            replace_worst(buffer, DistPair::new(rd, b));
            threshold = worst_distance(buffer);
            searcher.decrease_cutoff(threshold);
        }
    }

    // Sort descending so best (smallest distance) is at the end for pop().
    buffer.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
}

fn worst_distance<F: Float>(buffer: &[DistPair<F>]) -> F {
    buffer.iter().map(|n| n.distance).fold(F::neg_infinity(), |a, b| if a > b { a } else { b })
}

fn replace_worst<F: Float>(buffer: &mut [DistPair<F>], item: DistPair<F>) {
    if let Some(idx) = buffer
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
    {
        buffer[idx] = item;
    }
}

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

    use super::super::hdbscan_prim;
    use super::buffered_search_hdbscan;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

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

        let expected = hdbscan_prim(&data, 2).unwrap();
        let got = buffered_search_hdbscan(&tree, &data, 2, 1).unwrap();
        assert_eq!(got, expected);
    }
}
