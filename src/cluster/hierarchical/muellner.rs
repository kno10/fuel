use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::cluster::hierarchical::agnes::build_condensed_linkage_matrix;
use crate::cluster::hierarchical::anderberg::AnderbergState;
use crate::cluster::hierarchical::{Linkage, MergeHistory, idsize};
use crate::{DistanceData, Float};

#[derive(Clone, Copy, Debug)]
struct HeapEntry<F: Float> {
    dist: F,
    x: usize,
    y: usize,
}

impl<F: Float> PartialEq for HeapEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.x == other.x && self.y == other.y
    }
}

impl<F: Float> Eq for HeapEntry<F> {}

impl<F: Float> PartialOrd for HeapEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<F: Float> Ord for HeapEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => self.x.cmp(&other.x).then_with(|| self.y.cmp(&other.y)),
        }
    }
}

/// Perform hierarchical clustering using Müllner's generic-linkage approach
/// with an Anderberg nearest-neighbor cache and a heap for candidate retrieval.
#[must_use]
pub fn muellner<D, F: Float, L: Linkage<F> + Copy + Sync>(data: &D, linkage: L) -> MergeHistory<F>
where
    D: DistanceData<F> + Sync,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    let squared = data.is_squared_distance();
    let mat = build_condensed_linkage_matrix(data, linkage);
    let mut state = AnderbergState::new(mat, n);

    let mut heap = BinaryHeap::with_capacity(n);
    for x in 1..state.n() {
        push_candidate(&mut heap, &state.best, x);
    }

    for _ in 1..n {
        let (mindist, x, y) = pop_valid_merge(&mut heap, &state.best, &state.clustermap);
        let restored = linkage.restore(mindist, squared);
        let (size_x, size_y) = state.commit_merge(x, y, restored, usize::MAX);
        state.update_lw(linkage, mindist, x, y, size_x, size_y, |best, j| {
            push_candidate(&mut heap, best, j);
        });
        state.heights[y] = mindist;
        state.heights[x] = F::nan();
        state.refresh_best(y);
        push_candidate(&mut heap, &state.best, y);
    }

    state.builder.into_merges()
}

fn push_candidate<F: Float>(heap: &mut BinaryHeap<HeapEntry<F>>, best: &[(F, usize)], x: usize) {
    let y = best[x].1;
    if y != usize::MAX {
        heap.push(HeapEntry { dist: best[x].0, x, y });
    }
}

fn pop_valid_merge<F: Float>(
    heap: &mut BinaryHeap<HeapEntry<F>>, best: &[(F, usize)], clustermap: &[idsize],
) -> (F, usize, usize) {
    while let Some(entry) = heap.pop() {
        if entry.y == usize::MAX
            || clustermap[entry.x] == idsize::MAX
            || clustermap[entry.y] == idsize::MAX
        {
            continue;
        }
        if best[entry.x].1 != entry.y || best[entry.x].0 != entry.dist {
            continue;
        }
        return (entry.dist, entry.x.max(entry.y), entry.x.min(entry.y));
    }

    panic!("no merge candidate found");
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
mod tests {
    use super::muellner;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_condensed;
    use crate::cluster::hierarchical::{
        CentroidLinkage, CompleteLinkage, GroupAverageLinkage, MedianLinkage,
        MinimumSumSquaresLinkage, MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage,
        SingleLinkage, WardLinkage, WeightedAverageLinkage,
    };
    use crate::distance::{Euclidean, SquaredEuclidean};

    #[test]
    fn muellner_average_regression() {
        test_clustering_condensed("Muellner", "average", Euclidean, |condensed, min_clusters| {
            let history = muellner(condensed, GroupAverageLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn muellner_complete_regression() {
        test_clustering_condensed("Muellner", "complete", Euclidean, |condensed, min_clusters| {
            let history = muellner(condensed, CompleteLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn muellner_single_regression() {
        test_clustering_condensed("Muellner", "single", Euclidean, |condensed, min_clusters| {
            let history = muellner(condensed, SingleLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn muellner_ward_regression() {
        test_clustering_condensed(
            "Muellner",
            "ward",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = muellner(condensed, WardLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn muellner_weighted_average_regression() {
        test_clustering_condensed(
            "Muellner",
            "weighted_average",
            Euclidean,
            |condensed, min_clusters| {
                let history = muellner(condensed, WeightedAverageLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn muellner_centroid_regression() {
        test_clustering_condensed(
            "Muellner",
            "centroid",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = muellner(condensed, CentroidLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn muellner_median_regression() {
        test_clustering_condensed(
            "Muellner",
            "median",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = muellner(condensed, MedianLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn muellner_minimum_variance_increase_regression() {
        test_clustering_condensed("Muellner", "mivar", Euclidean, |condensed, min_clusters| {
            let history = muellner(condensed, MinimumVarianceIncreaseLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn muellner_minimum_sum_squares_regression() {
        test_clustering_condensed("Muellner", "mnssq", Euclidean, |condensed, min_clusters| {
            let history = muellner(condensed, MinimumSumSquaresLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn muellner_minimum_variance_regression() {
        test_clustering_condensed("Muellner", "mnvar", Euclidean, |condensed, min_clusters| {
            let history = muellner(condensed, MinimumVarianceLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }
}
