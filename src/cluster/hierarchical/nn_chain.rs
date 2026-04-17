//! Nearest-neighbor chain clustering on condensed distance matrices.
//!
//! This algorithm grows a chain of nearest neighbors and closes the chain when
//! it returns to a previous cluster.  It avoids scanning all active pairs for
//! every merge, which can make it faster for some linkages.
//!
//! On linkages that permit inversions, the NN-chain heuristic may produce a
//! different merge order than strict agglomerative clustering, because the
//! chain closure does not always select the globally minimal active pair.
//!
//! Results should therefore be compared carefully for linkages such as
//! `CentroidLinkage` and `MedianLinkage` that are not guaranteed to be
//! reducible.

use crate::cluster::hierarchical::agnes::build_condensed_linkage_matrix;
use crate::cluster::hierarchical::common::{
    condensed_get, condensed_set, find_active, shrink_active_end,
};
use crate::cluster::hierarchical::{Builder, Linkage, MergeHistory, idsize};
use crate::{DistanceData, Float};

/// Perform hierarchical clustering using the NN-Chain algorithm.
///
/// Input and output conventions are the same as [`crate::cluster::hierarchical::agnes`].
/// The input matrix uses lower-triangular condensed indexing.
#[must_use]
pub fn nn_chain<D, F: Float, L: Linkage<F> + Copy + Sync>(data: &D, linkage: L) -> MergeHistory<F>
where
    D: DistanceData<F> + Sync,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");

    let mut builder = Builder::<F>::new(n);
    let squared = data.is_squared_distance();
    let mut mat = build_condensed_linkage_matrix(data, linkage);
    let mut clustermap: Vec<idsize> = (0..(n as idsize)).collect();
    let mut heights = vec![F::zero(); n];
    let mut end = n;
    let mut chain: Vec<usize> = Vec::with_capacity((n / 4).max(2));
    let mut merged = 0usize;
    let mut warned_inversion = false;

    while merged < n - 1 {
        let mut a;
        let mut b;

        if chain.len() < 2 {
            a = find_active(0, end, &clustermap).expect("at least one active cluster");
            b = find_active(a + 1, end, &clustermap).expect("at least two active clusters");
            chain.clear();
            chain.push(a);
        } else {
            (a, b) = (chain[chain.len() - 2], chain[chain.len() - 1]);
            if clustermap[a] == idsize::MAX {
                if !warned_inversion {
                    eprintln!(
                        "Detected an inversion in the clustering. NNChain on irreducible linkages may yield different results."
                    );
                    warned_inversion = true;
                }
                chain.truncate(chain.len().saturating_sub(2));
                continue;
            }
            debug_assert!(clustermap[b] != idsize::MAX);
            chain.pop();
        }

        let mut min_dist = condensed_get(&mat, a, b);
        loop {
            let mut c = b;
            for (i, opt) in clustermap.iter().enumerate().take(end) {
                if i == a || i == b || *opt == idsize::MAX {
                    continue;
                }
                let d = condensed_get(&mat, a, i);
                if d < min_dist {
                    min_dist = d;
                    c = i;
                }
            }
            (b, a) = (a, c);
            chain.push(a);
            if chain.len() >= 3 && a == chain[chain.len() - 3] {
                break;
            }
        }

        let (x, y) = if a > b { (a, b) } else { (b, a) };
        let (cx, cy) = (clustermap[x], clustermap[y]);
        let (size_x, size_y) = (builder.get_size(cx), builder.get_size(cy));

        let new_id = builder.add(cx.min(cy), linkage.restore(min_dist, squared), cy.max(cx));
        clustermap[y] = new_id;
        clustermap[x] = idsize::MAX;

        let height_x = heights[x];
        let height_y = heights[y];

        for (j, opt) in clustermap.iter().enumerate().take(end) {
            if j == x || j == y || *opt == idsize::MAX {
                continue;
            }
            let (d_xj, d_yj) = (condensed_get(&mat, x, j), condensed_get(&mat, y, j));
            let size_j = builder.get_size(clustermap[j]);
            let height_j = heights[j];
            let d = linkage.combine(
                size_x, d_xj, size_y, d_yj, size_j, min_dist, height_x, height_y, height_j,
            );
            condensed_set(&mut mat, y, j, d);
        }

        heights[y] = min_dist;
        heights[x] = F::nan();

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

#[cfg(test)]
mod tests {
    use super::nn_chain;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_condensed;
    use crate::cluster::hierarchical::{
        CentroidLinkage, CompleteLinkage, GroupAverageLinkage, MedianLinkage,
        MinimumSumSquaresLinkage, MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage,
        SingleLinkage, WardLinkage, WeightedAverageLinkage,
    };
    use crate::distance::{Euclidean, SquaredEuclidean};

    #[test]
    fn nn_chain_average_regression() {
        test_clustering_condensed("NNChain", "average", Euclidean, |condensed, min_clusters| {
            let history = nn_chain(condensed, GroupAverageLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn nn_chain_complete_regression() {
        test_clustering_condensed("NNChain", "complete", Euclidean, |condensed, min_clusters| {
            let history = nn_chain(condensed, CompleteLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn nn_chain_single_regression() {
        test_clustering_condensed("NNChain", "single", Euclidean, |condensed, min_clusters| {
            let history = nn_chain(condensed, SingleLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn nn_chain_ward_regression() {
        test_clustering_condensed(
            "NNChain",
            "ward",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = nn_chain(condensed, WardLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn nn_chain_minimum_variance_increase_regression() {
        test_clustering_condensed("NNChain", "mivar", Euclidean, |condensed, min_clusters| {
            let history = nn_chain(condensed, MinimumVarianceIncreaseLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn nn_chain_minimum_sum_squares_regression() {
        test_clustering_condensed("NNChain", "mnssq", Euclidean, |condensed, min_clusters| {
            let history = nn_chain(condensed, MinimumSumSquaresLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn nn_chain_minimum_variance_regression() {
        test_clustering_condensed("NNChain", "mnvar", Euclidean, |condensed, min_clusters| {
            let history = nn_chain(condensed, MinimumVarianceLinkage);
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn nn_chain_centroid_regression() {
        test_clustering_condensed(
            "NNChain",
            "centroid",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = nn_chain(condensed, CentroidLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn nn_chain_median_regression() {
        test_clustering_condensed(
            "NNChain",
            "median",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = nn_chain(condensed, MedianLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn nn_chain_weighted_average_regression() {
        test_clustering_condensed(
            "NNChain",
            "weighted_average",
            Euclidean,
            |condensed, min_clusters| {
                let history = nn_chain(condensed, WeightedAverageLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }
}
