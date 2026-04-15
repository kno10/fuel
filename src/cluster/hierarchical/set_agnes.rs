use crate::cluster::hierarchical::common::{
    initialize_set_clusters, set_find_best_active_pair, shrink_active_end, update_set_entry,
};
use crate::cluster::hierarchical::{Builder, MergeHistory, SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Generic agglomerative clustering routine parameterised by a `SetLinkage`
/// implementation.
#[must_use]
pub fn set_agnes<D, L, F, S>(data: &D) -> MergeHistory<F>
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
    let mut end = n;

    for _ in 1..n {
        let (x, y, mindist) = set_find_best_active_pair::<F>(&distances, &clustermap, end);
        let cx = std::mem::take(&mut members[x]);
        let (summary_x, summary_y) = (&summaries[x], &summaries[y]);
        let (_, merged_summary) = L::cluster_distance(data, summary_x, summary_y, &cx, &members[y]);
        let proto = L::merged_prototype(&merged_summary);

        let (xx, yy) = (clustermap[x], clustermap[y]);
        let distance = L::restore(mindist, data.is_squared_distance());
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
    }

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
    for (j, opt) in clustermap.iter().enumerate().take(c) {
        if *opt == idsize::MAX {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, distances, members, summaries, c, j);
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
    use super::set_agnes;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::cluster::hierarchical::{
        CompleteLinkage, GroupAverageLinkage, HausdorffLinkage, MedoidLinkage, MinimaxLinkage,
        MinimumSumIncreaseLinkage, MinimumSumLinkage, MinimumSumSquaresLinkage,
        MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage, SingleLinkage, WardLinkage,
    };

    #[test]
    fn set_agnes_minimax_regression() {
        test_clustering_table(
            "SetAGNES",
            "minimax",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MinimaxLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_single_regression() {
        test_clustering_table(
            "SetAGNES",
            "single",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, SingleLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_complete_regression() {
        test_clustering_table(
            "SetAGNES",
            "complete",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, CompleteLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_group_average_regression() {
        test_clustering_table(
            "SetAGNES",
            "average",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, GroupAverageLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_ward_regression() {
        test_clustering_table(
            "SetAGNES",
            "ward",
            crate::distance::SquaredEuclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, WardLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_hausdorff_regression() {
        test_clustering_table(
            "SetAGNES",
            "hausdorff",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, HausdorffLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_medoid_regression() {
        test_clustering_table(
            "SetAGNES",
            "medoid",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MedoidLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_minimum_variance_increase_regression() {
        test_clustering_table(
            "SetAGNES",
            "mivar",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MinimumVarianceIncreaseLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_minimum_sum_squares_regression() {
        test_clustering_table(
            "SetAGNES",
            "mnssq",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MinimumSumSquaresLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_minimum_variance_regression() {
        test_clustering_table(
            "SetAGNES",
            "mnvar",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MinimumVarianceLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_minimum_sum_regression() {
        test_clustering_table(
            "SetAGNES",
            "minimum_sum",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MinimumSumLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_agnes_minimum_sum_increase_regression() {
        test_clustering_table(
            "SetAGNES",
            "minimum_sum_increase",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_agnes::<_, MinimumSumIncreaseLinkage, _, _>(access);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
