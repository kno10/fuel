//! Set-based Müllner clustering with heap-backed candidate selection.
//!
//! This implementation combines the set-based `SetLinkage` abstraction with the
//! `Müllner` heap-backed candidate retrieval strategy.  It uses explicit cluster
//! membership summaries while retaining the accelerated nearest-neighbor cache
//! and priority heap used by `Müllner`.

use crate::cluster::hierarchical::anderberg::AnderbergState;
use crate::cluster::hierarchical::common::{
    initialize_set_clusters, set_update_cache, triangle_index, update_set_entry,
};
use crate::cluster::hierarchical::{MergeHistory, SetLinkage, idsize};
use crate::{CandidateHeap, DistPair, DistanceData, Float};

/// Perform set-based Müllner hierarchical clustering.
pub fn set_muellner<D, L, F, S>(data: &D) -> Result<MergeHistory<F>, String>
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
    let mut heap = CandidateHeap::<F>::with_capacity(n);
    for x in 1..state.n() {
        push_candidate(&mut heap, &state.best, x as idsize);
    }

    for _ in 1..n {
        crate::poll_interrupted()?;
        let (mindist, x, y) = pop_valid_merge(&mut heap, &state.best, &state.clustermap);
        let x = x as usize;
        let y = y as usize;
        let cx = std::mem::take(&mut members[x]);
        let (_, merged_summary) =
            L::cluster_distance(data, &summaries[x], &summaries[y], &cx, &members[y]);
        let restored = L::restore(mindist, data.is_squared_distance());
        state.commit_merge(x, y, restored, L::merged_prototype(&merged_summary));
        members[y].extend(cx);
        summaries[y] = merged_summary;
        update_matrices::<D, L, F, S>(data, &mut state, &mut heap, &members, &summaries, x, y);
        state.refresh_best(y);
        push_candidate(&mut heap, &state.best, y as idsize);
    }

    Ok(state.builder.into_merges())
}

fn push_candidate<F: Float>(heap: &mut CandidateHeap<F>, best: &[(F, idsize)], x: idsize) {
    let y = best[x as usize].1;
    if y != idsize::MAX {
        heap.push(DistPair::new(best[x as usize].0, x as usize));
    }
}

fn pop_valid_merge<F: Float>(
    heap: &mut CandidateHeap<F>, best: &[(F, idsize)], clustermap: &[idsize],
) -> (F, idsize, idsize) {
    while let Some(entry) = heap.pop() {
        let x = entry.index as idsize;
        if clustermap[x as usize] == idsize::MAX {
            continue;
        }
        if best[x as usize].0 != entry.distance {
            continue;
        }
        let y = best[x as usize].1;
        if y == idsize::MAX || clustermap[y as usize] == idsize::MAX {
            continue;
        }
        return (entry.distance, x.max(y), x.min(y));
    }

    panic!("no merge candidate found");
}

fn update_matrices<D, L, F, S>(
    data: &D, state: &mut AnderbergState<F>, heap: &mut CandidateHeap<F>, members: &[Vec<idsize>],
    summaries: &[S], x: usize, y: usize,
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
            if set_update_cache::<F>(
                &state.mat,
                &state.clustermap,
                &mut state.best,
                x,
                y,
                b,
                state.mat[yoffset + b],
            ) {
                push_candidate(heap, &state.best, b as idsize);
            }
        }
    }

    for a in (y + 1)..state.end {
        if state.clustermap[a] == idsize::MAX {
            continue;
        }
        update_set_entry::<D, L, F, S>(data, &mut state.mat, members, summaries, a, y);
        let d = state.mat[triangle_index(a, y)];
        if set_update_cache::<F>(&state.mat, &state.clustermap, &mut state.best, x, y, a, d) {
            push_candidate(heap, &state.best, a as idsize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::set_muellner;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::cluster::hierarchical::{
        GroupAverageLinkage, MedoidLinkage, MinimaxLinkage, MinimumSumSquaresLinkage,
        MinimumVarianceLinkage, SingleLinkage, WardLinkage,
    };

    #[test]
    fn set_muellner_group_average_regression() {
        test_clustering_table(
            "SetMuellner",
            "average",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_muellner::<_, GroupAverageLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_muellner_minimax_regression() {
        test_clustering_table(
            "SetMuellner",
            "minimax",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_muellner::<_, MinimaxLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_muellner_medoid_regression() {
        test_clustering_table(
            "SetMuellner",
            "medoid",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_muellner::<_, MedoidLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_muellner_single_regression() {
        test_clustering_table(
            "SetMuellner",
            "single",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_muellner::<_, SingleLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_muellner_ward_regression() {
        test_clustering_table(
            "SetMuellner",
            "ward",
            crate::distance::SquaredEuclidean,
            |access, min_clusters| {
                let history = set_muellner::<_, WardLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_muellner_minimum_variance_regression() {
        test_clustering_table(
            "SetMuellner",
            "mnvar",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_muellner::<_, MinimumVarianceLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn set_muellner_minimum_sum_squares_regression() {
        test_clustering_table(
            "SetMuellner",
            "mnssq",
            crate::distance::Euclidean,
            |access, min_clusters| {
                let history = set_muellner::<_, MinimumSumSquaresLinkage, _, _>(access).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
