use crate::cluster::hierarchical::common::{
    condensed_get, find_active, initialize_set_clusters, shrink_active_end, update_set_entry,
};
use crate::cluster::hierarchical::{Builder, MergeHistory, SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Nearest‑neighbor chain heuristic agglomeration using an arbitrary set‑based
/// linkage criterion.
#[must_use]
pub fn set_nn_chain<D, L, F, S>(data: &D) -> MergeHistory<F>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return MergeHistory::new();
    }

    let (mut members, mut summaries, mut distances, mut clustermap) =
        initialize_set_clusters::<D, L, F, S>(data);
    let mut builder = Builder::<F>::new(n);
    let mut chain: Vec<usize> = Vec::with_capacity((n / 4).max(2));
    let mut end = n;
    let mut merged = 0usize;

    while merged < n - 1 {
        let mut a;
        let mut b;
        if chain.len() < 2 {
            a = find_active(0, end, &clustermap).expect("at least one active cluster");
            b = find_active(a + 1, end, &clustermap).expect("at least two active clusters");
            chain.clear();
            chain.push(a);
        } else {
            a = chain[chain.len() - 2];
            b = chain[chain.len() - 1];
            if clustermap[a] == idsize::MAX {
                chain.truncate(chain.len().saturating_sub(2));
                continue;
            }
            debug_assert!(clustermap[b] != idsize::MAX);
            chain.pop();
        }

        let mut min_dist = condensed_get(&distances, a, b);
        loop {
            let mut c = b;
            for (i, opt) in clustermap.iter().enumerate().take(end) {
                if i == a || i == b || *opt == idsize::MAX {
                    continue;
                }
                let raw = condensed_get(&distances, a, i);
                if raw < min_dist {
                    min_dist = raw;
                    c = i;
                }
            }
            b = a;
            a = c;
            chain.push(a);
            if chain.len() >= 3 && a == chain[chain.len() - 3] {
                break;
            }
        }

        let (x, y) = if a > b { (a, b) } else { (b, a) };
        let cx = std::mem::take(&mut members[x]);
        let (_, merged_summary) =
            L::cluster_distance(data, &summaries[x], &summaries[y], &cx, &members[y]);
        let proto = L::merged_prototype(&merged_summary);

        let (xx, yy) = (clustermap[x], clustermap[y]);
        let distance = L::restore(min_dist, data.is_squared_distance());
        let new_id = builder.add_with_prototype(xx.min(yy), distance, yy.max(xx), proto);
        clustermap[y] = new_id;
        clustermap[x] = idsize::MAX;

        members[y].extend(cx);
        summaries[y] = merged_summary;

        update_matrices::<D, L, F, S>(
            data,
            &mut distances,
            &clustermap,
            &members,
            &summaries,
            y,
            end,
        );

        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }

        if chain.len() >= 3 {
            chain.truncate(chain.len() - 3);
        } else {
            chain.clear();
        }
        chain.push(y);
        merged += 1;
    }

    builder.optimize_order_in_place();
    builder.into_merges()
}

fn update_matrices<D, L, F, S>(
    data: &D, distances: &mut [F], clustermap: &[idsize], members: &[Vec<idsize>], summaries: &[S],
    c: usize, end: usize,
) where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    for (y, opt) in clustermap.iter().enumerate().take(c) {
        if *opt == idsize::MAX {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, distances, members, summaries, c, y);
    }
    for (x, opt) in clustermap.iter().enumerate().skip(c + 1).take(end - (c + 1)) {
        if *opt == idsize::MAX {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, distances, members, summaries, x, c);
    }
}

#[cfg(test)]
mod tests {
    use super::set_nn_chain;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::cluster::hierarchical::{
        CompleteLinkage, GroupAverageLinkage, HausdorffLinkage, MedoidLinkage, MinimaxLinkage,
        MinimumSumIncreaseLinkage, MinimumSumLinkage, MinimumSumSquaresLinkage,
        MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage, SingleLinkage, WardLinkage,
    };

    #[test]
    fn set_nn_chain_minimax_regression() {
        test_clustering_table(
            "SetNNChain",
            "minimax",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, MinimaxLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_group_average_regression() {
        test_clustering_table(
            "SetNNChain",
            "average",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, GroupAverageLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_complete_regression() {
        test_clustering_table(
            "SetNNChain",
            "complete",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, CompleteLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_single_regression() {
        test_clustering_table(
            "SetNNChain",
            "single",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, SingleLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_ward_regression() {
        test_clustering_table(
            "SetNNChain",
            "ward",
            crate::distance::SquaredEuclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, WardLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_hausdorff_regression() {
        test_clustering_table(
            "SetNNChain",
            "hausdorff",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, HausdorffLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_medoid_regression() {
        test_clustering_table(
            "SetNNChain",
            "medoid",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, MedoidLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_minimum_variance_increase_regression() {
        test_clustering_table(
            "SetNNChain",
            "mivar",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, MinimumVarianceIncreaseLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_minimum_sum_squares_regression() {
        test_clustering_table(
            "SetNNChain",
            "mnssq",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, MinimumSumSquaresLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_minimum_variance_regression() {
        test_clustering_table(
            "SetNNChain",
            "mnvar",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, MinimumVarianceLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_minimum_sum_regression() {
        test_clustering_table(
            "SetNNChain",
            "minimum_sum",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, MinimumSumLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_nn_chain_minimum_sum_increase_regression() {
        test_clustering_table(
            "SetNNChain",
            "minimum_sum_increase",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_nn_chain::<_, MinimumSumIncreaseLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
