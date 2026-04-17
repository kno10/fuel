//! Anderberg's nearest-neighbor accelerated agglomerative clustering.
//!
//! This method is a nearest-neighbor cache specialization of `AGNES`.  It
//! maintains a current best neighbor for each active cluster and updates only
//! the affected entries after each merge.
//!
//! Compared to plain `AGNES`, it avoids repeated full scans of the condensed
//! distance matrix, but it still relies on an explicit stored matrix and the
//! generic Lance-Williams recurrence.

use crate::cluster::hierarchical::agnes::build_condensed_linkage_matrix;
use crate::cluster::hierarchical::common::{
    condensed_get, condensed_set, shrink_active_end, triangle_index,
};
use crate::cluster::hierarchical::{Builder, Linkage, MergeHistory, idsize};
use crate::{DistanceData, Float};

fn find_best<F: Float>(distances: &[F], clustermap: &[idsize], best: &mut [(F, idsize)], j: usize) {
    let mut best_pair = (F::infinity(), idsize::MAX);
    for i in 0..j {
        if clustermap[i] == idsize::MAX {
            continue;
        }
        let d = distances[triangle_index(j, i)];
        if d <= best_pair.0 {
            best_pair = (d, i as idsize);
        }
    }
    best[j] = best_pair;
}

fn initialize_nn_cache<F: Float>(distances: &[F], clustermap: &[idsize], best: &mut [(F, idsize)]) {
    for x in 1..best.len() {
        if clustermap[x] != idsize::MAX {
            find_best(distances, clustermap, best, x);
        }
    }
}

/// Working state shared by all Anderberg-family algorithms.
pub(crate) struct AnderbergState<F: Float> {
    pub(crate) mat: Vec<F>,
    pub(crate) clustermap: Vec<idsize>,
    pub(crate) heights: Vec<F>,
    pub(crate) builder: Builder<F>,
    pub(crate) best: Vec<(F, idsize)>,
    pub(crate) end: usize,
}

impl<F: Float> AnderbergState<F> {
    pub(crate) fn new(mat: Vec<F>, n: usize) -> Self {
        let clustermap: Vec<idsize> = (0..(n as idsize)).collect();
        let heights = vec![F::zero(); n];
        let builder = Builder::<F>::new(n);
        let mut best = vec![(F::infinity(), idsize::MAX); n];
        initialize_nn_cache(&mat, &clustermap, &mut best);
        Self { mat, clustermap, heights, builder, best, end: n }
    }

    pub(crate) fn n(&self) -> usize { self.best.len() }

    /// Scan for the best merge candidate. Returns `(dist, x, y)` with `x > y`.
    pub(crate) fn find_merge(&self) -> (F, usize, usize) {
        let mut be = (F::infinity(), usize::MAX, usize::MAX);
        for cx in 1..self.end {
            if self.clustermap[cx] == idsize::MAX || self.best[cx].1 == idsize::MAX {
                continue;
            }
            let cy = self.best[cx].1 as usize;
            if self.best[cx].0 <= be.0 {
                be = (self.best[cx].0, cx, cy);
            }
        }
        assert!(be.1 != usize::MAX, "no merge candidate found");
        (be.0, be.1.max(be.2), be.1.min(be.2))
    }

    /// Record the merge of clusters at positions x and y.
    /// Returns `(size_x, size_y)` - cluster sizes before the merge.
    pub(crate) fn commit_merge(
        &mut self, x: usize, y: usize, dist: F, prototype: usize,
    ) -> (usize, usize) {
        let (cid_x, cid_y) = (self.clustermap[x], self.clustermap[y]);
        let (size_x, size_y) = (self.builder.get_size(cid_x), self.builder.get_size(cid_y));
        let (a, b) = if cid_y <= cid_x { (cid_y, cid_x) } else { (cid_x, cid_y) };
        self.clustermap[y] = self.builder.add_with_prototype(a, dist, b, prototype);
        self.clustermap[x] = idsize::MAX;
        self.best[x] = (F::infinity(), idsize::MAX);
        if x == self.end - 1 {
            shrink_active_end(&self.clustermap, &mut self.end);
        }
        (size_x, size_y)
    }

    /// Apply the Lance-Williams update to the distance matrix and NN cache.
    /// Calls `on_update(best, j)` for each j whose cache entry changes.
    pub(crate) fn update_lw<L, OnUpdate>(
        &mut self, linkage: L, mindist: F, x: usize, y: usize, size_x: usize, size_y: usize,
        mut on_update: OnUpdate,
    ) where
        L: Linkage<F> + Copy,
        OnUpdate: FnMut(&mut [(F, idsize)], usize),
    {
        for j in 0..self.end {
            if j == x || j == y || self.clustermap[j] == idsize::MAX {
                continue;
            }
            let d_xj = condensed_get(&self.mat, x, j);
            let d_yj = condensed_get(&self.mat, y, j);
            let size_j = self.builder.get_size(self.clustermap[j]);
            let height_x = self.heights[x];
            let height_y = self.heights[y];
            let height_j = self.heights[j];
            let d = linkage
                .combine(size_x, d_xj, size_y, d_yj, size_j, mindist, height_x, height_y, height_j);
            condensed_set(&mut self.mat, y, j, d);
            if self.update_best(x, y, j, d) {
                on_update(&mut self.best, j);
            }
        }
    }

    fn update_best(&mut self, x: usize, y: usize, j: usize, d: F) -> bool {
        if y < j && d <= self.best[j].0 {
            self.best[j] = (d, y as idsize);
            return true;
        }
        if self.best[j].1 == x as idsize || self.best[j].1 == y as idsize {
            let old = self.best[j];
            find_best(&self.mat, &self.clustermap, &mut self.best, j);
            return self.best[j] != old;
        }
        false
    }

    /// Refresh the NN cache entry for y (no-op if y == 0).
    pub(crate) fn refresh_best(&mut self, y: usize) {
        if y > 0 {
            find_best(&self.mat, &self.clustermap, &mut self.best, y);
        }
    }
}

/// Perform hierarchical clustering using Anderberg's NN-cache acceleration.
///
/// Input and output conventions are the same as [`crate::cluster::hierarchical::agnes`].
#[must_use]
pub fn anderberg<D, F: Float, L: Linkage<F> + Copy + Sync>(data: &D, linkage: L) -> MergeHistory<F>
where
    D: DistanceData<F> + Sync,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");
    let squared = data.is_squared_distance();
    let mat = build_condensed_linkage_matrix(data, linkage);
    let mut state = AnderbergState::new(mat, n);
    for _ in 1..n {
        let (mindist, x, y) = state.find_merge();
        let restored = linkage.restore(mindist, squared);
        let (size_x, size_y) = state.commit_merge(x, y, restored, usize::MAX);
        state.update_lw(linkage, mindist, x, y, size_x, size_y, |_, _| {});
        state.heights[y] = mindist;
        state.heights[x] = F::nan();
        state.refresh_best(y);
    }
    state.builder.into_merges()
}

#[cfg(test)]
mod tests {
    use super::anderberg;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_condensed;
    use crate::cluster::hierarchical::{
        CentroidLinkage, CompleteLinkage, GroupAverageLinkage, MedianLinkage,
        MinimumSumSquaresLinkage, MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage,
        SingleLinkage, WardLinkage, WeightedAverageLinkage,
    };
    use crate::distance::{Euclidean, SquaredEuclidean};

    #[test]
    fn anderberg_average_regression() {
        test_clustering_condensed("Anderberg", "average", Euclidean, |condensed, min_clusters| {
            let history = anderberg(condensed, GroupAverageLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn anderberg_complete_regression() {
        test_clustering_condensed("Anderberg", "complete", Euclidean, |condensed, min_clusters| {
            let history = anderberg(condensed, CompleteLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn anderberg_single_regression() {
        test_clustering_condensed("Anderberg", "single", Euclidean, |condensed, min_clusters| {
            let history = anderberg(condensed, SingleLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn anderberg_ward_regression() {
        test_clustering_condensed(
            "Anderberg",
            "ward",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = anderberg(condensed, WardLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn anderberg_centroid_regression() {
        test_clustering_condensed(
            "Anderberg",
            "centroid",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = anderberg(condensed, CentroidLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn anderberg_median_regression() {
        test_clustering_condensed(
            "Anderberg",
            "median",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = anderberg(condensed, MedianLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn anderberg_weighted_average_regression() {
        test_clustering_condensed(
            "Anderberg",
            "weighted_average",
            Euclidean,
            |condensed, min_clusters| {
                let history = anderberg(condensed, WeightedAverageLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn anderberg_minimum_variance_increase_regression() {
        test_clustering_condensed("Anderberg", "mivar", Euclidean, |condensed, min_clusters| {
            let history = anderberg(condensed, MinimumVarianceIncreaseLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn anderberg_minimum_sum_squares_regression() {
        test_clustering_condensed("Anderberg", "mnssq", Euclidean, |condensed, min_clusters| {
            let history = anderberg(condensed, MinimumSumSquaresLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn anderberg_minimum_variance_regression() {
        test_clustering_condensed("Anderberg", "mnvar", Euclidean, |condensed, min_clusters| {
            let history = anderberg(condensed, MinimumVarianceLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }
}
