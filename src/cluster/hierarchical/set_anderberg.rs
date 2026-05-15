//! Set-based Anderberg clustering with a nearest-neighbor cache.
//!
//! This method combines the set-based `SetLinkage` abstraction with the
//! Anderberg nearest-neighbor acceleration.  It is suitable for linkages that
//! require explicit cluster membership and summary statistics, while still
//! limiting the amount of work performed after each merge.

use crate::cluster::hierarchical::anderberg::AnderbergState;
use crate::cluster::hierarchical::common::{
    initialize_set_clusters, set_update_cache, triangle_index, update_set_entry,
};
use crate::cluster::hierarchical::{MergeHistory, SetLinkage, idsize};
use crate::{DistanceData, Float};

/// Perform set-based Anderberg hierarchical clustering.
pub fn set_anderberg<D, L, F, S>(data: &D) -> Result<MergeHistory<F>, String>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    if n == 1 {
        return Ok(MergeHistory::new());
    }

    let (mut members, mut summaries, distances, _) = initialize_set_clusters::<D, L, F, S>(data)?;
    let mut state = AnderbergState::new(distances, n);

    for _ in 1..n {
        crate::poll_interrupted()?;
        let (mindist, x, y) = state.find_merge();
        // Move members[x] out before the merge so we compute cluster_distance once.
        let cx = std::mem::take(&mut members[x]);
        let (_, merged_summary) =
            L::cluster_distance(data, &summaries[x], &summaries[y], &cx, &members[y]);
        let restored = L::restore(mindist, data.is_squared_distance());
        state.commit_merge(x, y, restored, L::merged_prototype(&merged_summary));
        members[y].extend(cx);
        summaries[y] = merged_summary;
        update_matrices::<D, L, F, S>(data, &mut state, &members, &summaries, x, y);
        state.refresh_best(y);
    }

    Ok(state.builder.into_merges())
}

fn update_matrices<D, L, F, S>(
    data: &D, state: &mut AnderbergState<F>, members: &[Vec<idsize>], summaries: &[S], x: usize,
    y: usize,
) where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    if y > 0 {
        let yoffset = triangle_index(y, 0);
        for b in 0..y {
            if state.clustermap[b] == idsize::MAX {
                continue;
            }
            update_set_entry::<D, L, F, S>(data, &mut state.mat, members, summaries, y, b);
            let _ = set_update_cache::<F>(
                &state.mat,
                &state.clustermap,
                &mut state.best,
                x,
                y,
                b,
                state.mat[yoffset + b],
            );
        }
    }

    for a in (y + 1)..state.end {
        if state.clustermap[a] == idsize::MAX {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, &mut state.mat, members, summaries, a, y);
        let d = state.mat[triangle_index(a, y)];
        let _ = set_update_cache::<F>(&state.mat, &state.clustermap, &mut state.best, x, y, a, d);
    }
}

#[cfg(test)]
mod tests {
    use super::set_anderberg;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::cluster::hierarchical::{
        CompleteLinkage, GroupAverageLinkage, HausdorffLinkage, MedoidLinkage, MinimaxLinkage,
        MinimumSumIncreaseLinkage, MinimumSumLinkage, MinimumSumSquaresLinkage,
        MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage, SingleLinkage, WardLinkage,
    };

    #[test]
    fn set_anderberg_minimax_regression() {
        test_clustering_table(
            "SetAnderberg",
            "minimax",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, MinimaxLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_group_average_regression() {
        test_clustering_table(
            "SetAnderberg",
            "average",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, GroupAverageLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_complete_regression() {
        test_clustering_table(
            "SetAnderberg",
            "complete",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, CompleteLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_single_regression() {
        test_clustering_table(
            "SetAnderberg",
            "single",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, SingleLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_ward_regression() {
        test_clustering_table(
            "SetAnderberg",
            "ward",
            crate::distance::SquaredEuclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, WardLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_hausdorff_regression() {
        test_clustering_table(
            "SetAnderberg",
            "hausdorff",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, HausdorffLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_medoid_regression() {
        test_clustering_table(
            "SetAnderberg",
            "medoid",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, MedoidLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_minimum_variance_increase_regression() {
        test_clustering_table(
            "SetAnderberg",
            "mivar",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, MinimumVarianceIncreaseLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_minimum_sum_squares_regression() {
        test_clustering_table(
            "SetAnderberg",
            "mnssq",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, MinimumSumSquaresLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_minimum_variance_regression() {
        test_clustering_table(
            "SetAnderberg",
            "mnvar",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, MinimumVarianceLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_minimum_sum_regression() {
        test_clustering_table(
            "SetAnderberg",
            "minimum_sum",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, MinimumSumLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_anderberg_minimum_sum_increase_regression() {
        test_clustering_table(
            "SetAnderberg",
            "minimum_sum_increase",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_anderberg::<_, MinimumSumIncreaseLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
