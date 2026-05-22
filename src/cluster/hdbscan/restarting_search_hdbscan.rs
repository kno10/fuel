use crate::api::DistanceData;
use crate::cluster::hdbscan::hdbscan_common::{HdbscanHierarchy, compute_core_distances_tree};
use crate::cluster::hierarchical::search_single_link_common::{ClusterBuilder, SameClusterFilter};
use crate::{
    CandidateHeap, DistPair, DistanceSearch, Float, IndexQuery, KnnSearch, PrioritySearcher,
    PrioritySearcherFactory,
};

/// Restarting-search HDBSCAN-RS (RSSL-style acceleration with VP-tree search).
pub fn restarting_search_hdbscan<'a, S, D, F>(
    tree: &'a S, data: &'a D, min_points: usize,
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

    let core_distances = compute_core_distances_tree(tree, data, min_points)?;

    let mut builder = ClusterBuilder::new(n);
    let mut primary = CandidateHeap::<F>::new();
    let mut buffers: Vec<DistPair<F>> = vec![DistPair::undefined(); n];
    let mut node_cluster = vec![u32::MAX; n];

    // create one searcher and reuse it for all refill operations
    let mut searcher = S::priority_searcher(tree);

    let mut query = data.query();

    // initial fill for each point
    for (a, buf) in buffers.iter_mut().enumerate().take(n) {
        if builder.cluster_size_of_point(a) > 1 {
            continue; // duplicate, merged already
        }
        query.set_index(a);
        refill_neighbors(
            &mut builder,
            a,
            &query,
            buf,
            &mut searcher,
            &core_distances,
            &mut node_cluster,
        );
        if !buf.is_sentinel() {
            primary.push(DistPair::new(buf.distance, a));
        }
    }

    while builder.merge_count() < n - 1 {
        crate::poll_interrupted()?;
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
        if builder.merge_points(a, b, best_dist).is_some() && builder.merge_count() == n - 1 {
            break;
        }

        query.set_index(a);
        refill_neighbors(
            &mut builder,
            a,
            &query,
            buf,
            &mut searcher,
            &core_distances,
            &mut node_cluster,
        );

        if !buf.is_sentinel() {
            primary.push(DistPair::new(buf.distance, a));
        }
    }

    Ok(HdbscanHierarchy::new(builder.into_history(), core_distances))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refill_neighbors<F: Float, Q, S>(
    builder: &mut ClusterBuilder<F>, query_index: usize, query: &Q, buffer: &mut DistPair<F>,
    searcher: &mut S, core_distances: &[F], node_cluster: &mut [u32],
) where
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcher<F, Q>,
{
    searcher.reset();

    let cd = core_distances[query_index];
    let mut threshold = F::infinity();
    let query_component = builder.find(query_index);
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next_with_filter(
            query,
            &mut SameClusterFilter { builder, query_component, node_cluster },
        ) else {
            break;
        };
        let b = cand.index;
        let d = cand.distance;
        let rd = cd.max(core_distances[b]).max(d); // mutual reachability
        if rd < threshold {
            *buffer = DistPair::new(rd, b);
            threshold = rd;
            searcher.decrease_cutoff(rd);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    use super::restarting_search_hdbscan;
    use crate::TableWithDistance;
    use crate::api::Data;
    use crate::cluster::hdbscan::extraction::extract_clusters_with_noise;
    use crate::cluster::hdbscan::{buffered_search_hdbscan, hdbscan_prim};
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    #[test]
    fn restarting_search_hdbscan_matches_linear_mst() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 1.2],
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(11);
        let tree = VPTree::<f64>::new(&data, 3, &mut rng);

        let expected = hdbscan_prim(&data, 2).unwrap();
        let got = restarting_search_hdbscan(&tree, &data, 2).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn restarting_equals_buffered_random() {
        // RNG-based regression: ensure both variants produce the same *clustering*.
        let mut rng = StdRng::seed_from_u64(42);
        let points: Vec<Vec<f64>> =
            (0..30).map(|_| vec![rng.random::<f64>(), rng.random::<f64>()]).collect();

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree = VPTree::<f64>::new(&data, 3, &mut rng);

        let hist_r = restarting_search_hdbscan(&tree, &data, 2).unwrap();
        let hist_b = buffered_search_hdbscan(&tree, &data, 2, 1).unwrap();

        let labels_r = extract_clusters_with_noise(&hist_r.merges, data.len(), 2);
        let labels_b = extract_clusters_with_noise(&hist_b.merges, data.len(), 2);
        assert_eq!(labels_r, labels_b);
    }
}
