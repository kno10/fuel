//! Agglomerative hierarchical clustering (AGNES) ported from the old Java
//! code base.  The algorithm works on a *condensed* distance matrix (the
//! strictly triangular part, stored as a contiguous slice) and supports a
//! variety of Lance-Williams linkage criteria.
//!
//! This is the baseline stored-matrix implementation in the library.  It
//! performs the generic O(n^3) merge process over the full condensed matrix
//! and is the reference against which NN-cache and heap-based optimizations
//! are compared.
//!
//! The output format is the same as `scipy.cluster.hierarchy.linkage`: a
//! sequence of `(i, j, dist, size)` quadruples where `i` and `j` are cluster
//! identifiers (original points have ids `0..n-1`, newly merged clusters are
//! assigned `n..` in merge order), `dist` is the merge distance, and `size` is
//! the number of original points in the newly created cluster.
//!
//! The input matrix is assumed to encode distances for pairs `(p,q)` with
//! `0 <= q < p < n`.  The index of such a pair in the slice is
//! `p*(p-1)/2 + q` which corresponds to the lower-triangular ordering used by
//! the Java implementation.  This is equivalent to the condensed form used by
//! SciPy (`pdist` output) except that SciPy uses the upper triangle; users can
//! simply transpose their indices to convert between the two representations.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::cluster::hierarchical::common::{shrink_active_end, triangle_index};
use crate::cluster::hierarchical::{Builder, Linkage, MergeHistory, idsize};
use crate::{DistanceData, Float};

/// Build a condensed linkage matrix from pairwise distances.
///
/// The returned vector has length `n*(n-1)/2` and stores the lower-triangular
/// distances for pairs `(x, y)` with `0 <= y < x < n`.
#[cfg(feature = "parallel")]
pub(crate) fn build_condensed_linkage_matrix<D, F: Float, L: Linkage<F> + Copy + Sync>(
    data: &D, linkage: L,
) -> Result<Vec<F>, String>
where
    D: DistanceData<F> + Sync,
{
    use std::sync::atomic::Ordering::Relaxed;
    let n = data.len();
    let squared = data.is_squared_distance();

    let rows = (1..n)
        .into_par_iter()
        .map(|x| {
            if crate::SHUTDOWN_REQUESTED.load(Relaxed) {
                return Err("interrupted".to_string());
            }
            Ok((0..x).map(|y| linkage.initial(data.distance(x, y), squared)).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<Vec<F>>, String>>()?;
    Ok(rows.into_iter().flatten().collect())
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn build_condensed_linkage_matrix<D, F: Float, L: Linkage<F> + Copy>(
    data: &D, linkage: L,
) -> Result<Vec<F>, String>
where
    D: DistanceData<F>,
{
    let n = data.len();
    let squared = data.is_squared_distance();
    let mut mat = Vec::with_capacity(n * (n - 1) / 2);
    for x in 1..n {
        crate::poll_interrupted()?;
        for y in 0..x {
            mat.push(linkage.initial(data.distance(x, y), squared));
        }
    }
    Ok(mat)
}

/// Perform the AGNES algorithm on a condensed lower-triangular distance
/// matrix.
///
/// - `distances` must have length `n*(n-1)/2` and encode the pairwise
///   distances for `(p,q)` with `0 <= q < p < n` in row-major order
///   (`triangle_index(p,q)`).
/// - `n` is the number of original objects.
/// - `linkage` selects the Lance-Williams linkage criterion to use.
///
/// Returns a `MergeHistory` in `SciPy` linkage format.
///
/// # Panics
///
/// * if the number of points `n` is zero
/// * if `distances.len()` does not equal `n*(n-1)/2`
///
/// The function converts the provided condensed matrix into an agglomerative
/// merge history and will `panic!` when the preconditions above are violated.
pub fn agnes<D, F: Float, L: Linkage<F> + Copy + Sync>(
    data: &D, linkage: L,
) -> Result<MergeHistory<F>, String>
where
    D: DistanceData<F> + Sync,
{
    let n = data.len();
    assert!(n > 0, "number of points must be positive");

    let mut builder = Builder::<F>::new(n);
    let squared = data.is_squared_distance();
    let mut mat = build_condensed_linkage_matrix(data, linkage)?;
    let mut clustermap: Vec<idsize> = (0..(n as idsize)).collect();
    let mut heights = vec![F::zero(); n];
    let mut end = n; // active end, we use shrinking

    // repeatedly merge until one cluster remains
    for _step in 1..n {
        crate::poll_interrupted()?;
        // find the closest pair among active objects
        let (mut mindist, mut x, mut y) = (F::infinity(), 0, 0);

        for ox in 0..end {
            if clustermap[ox] == idsize::MAX {
                continue;
            }
            for oy in 0..ox {
                if clustermap[oy] == idsize::MAX {
                    continue;
                }
                let d = mat[triangle_index(ox, oy)];
                // prefer the first occurrence of the minimum distance (not the
                // last) to match SciPy's behaviour.  using `<` rather than `<=`
                // achieves that because the loops scan from small indices
                // upward.
                if d < mindist {
                    (mindist, x, y) = (d, ox, oy);
                }
            }
        }

        debug_assert!(
            mindist.is_finite(),
            "AGNES found no merge candidate end={} active={:?}",
            end,
            clustermap[..end]
                .iter()
                .enumerate()
                .filter(|&(_, id)| *id != idsize::MAX)
                .collect::<Vec<_>>(),
        );

        // perform merge of (x,y) with y < x by construction of the loop above
        let (cid_x, cid_y) = (clustermap[x], clustermap[y]);
        debug_assert!(
            cid_x != idsize::MAX && cid_y != idsize::MAX,
            "AGNES selected inactive cluster x={} cid_x={} y={} cid_y={}",
            x,
            cid_x,
            y,
            cid_y,
        );
        let (size_x, size_y) = (builder.get_size(cid_x), builder.get_size(cid_y));

        // create new cluster id (keep y and drop x).  force the smaller
        // index to appear first in the history record so that our output
        // mirrors SciPy's convention.  restore the distance before storing.
        let newcid =
            builder.add(cid_x.min(cid_y), linkage.restore(mindist, squared), cid_x.max(cid_y));
        // note: even though we sorted (a,b) above, we still store the new
        // cluster in position `y` so that the distance update logic remains
        // correct (we always drop `x`).
        clustermap[y] = newcid;
        clustermap[x] = idsize::MAX; // deactivate

        let height_x = heights[x];
        let height_y = heights[y];

        // update distances in the matrix
        for j in 0..end {
            if j == x || j == y || clustermap[j] == idsize::MAX {
                continue;
            }
            // compute current distances from j to x and y
            let height_j = heights[j];
            let combined = linkage.combine(
                size_x,
                mat[triangle_index(x, j)],
                size_y,
                mat[triangle_index(y, j)],
                builder.get_size(clustermap[j]),
                mindist,
                height_x,
                height_y,
                height_j,
            );
            // triangle_index will handle the ordering of (y,j) internally
            mat[triangle_index(y, j)] = combined;
        }

        heights[y] = mindist;
        heights[x] = F::nan();

        // shrink active set if tail objects have disappeared
        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }
    }

    Ok(builder.into_merges())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::linkage::flexible_beta::FlexibleBetaLinkage;
    use crate::cluster::hierarchical::test::test_clustering_condensed;
    use crate::cluster::hierarchical::{
        CentroidLinkage, CompleteLinkage, GroupAverageLinkage, MedianLinkage,
        MinimumSumSquaresLinkage, MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage,
        SingleLinkage, WardLinkage, WeightedAverageLinkage,
    };
    use crate::distance::{Euclidean, SquaredEuclidean};

    #[test]
    fn agnes_group_average_regression() {
        test_clustering_condensed("AGNES", "average", Euclidean, |condensed, min_clusters| {
            let history = agnes(condensed, GroupAverageLinkage).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn agnes_weighted_average_regression() {
        test_clustering_condensed(
            "AGNES",
            "weighted_average",
            Euclidean,
            |condensed, min_clusters| {
                let history = agnes(condensed, WeightedAverageLinkage).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn agnes_centroid_regression() {
        test_clustering_condensed(
            "AGNES",
            "centroid",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = agnes(condensed, CentroidLinkage).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn agnes_median_regression() {
        test_clustering_condensed(
            "AGNES",
            "median",
            SquaredEuclidean,
            |condensed, min_clusters| {
                let history = agnes(condensed, MedianLinkage).unwrap();
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn agnes_complete_regression() {
        test_clustering_condensed("AGNES", "complete", Euclidean, |condensed, min_clusters| {
            let history = agnes(condensed, CompleteLinkage).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn agnes_single_regression() {
        test_clustering_condensed("AGNES", "single", Euclidean, |condensed, min_clusters| {
            let history = agnes(condensed, SingleLinkage).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn agnes_flexible_beta_regression() {
        test_clustering_condensed("AGNES", "flexible", Euclidean, |condensed, min_clusters| {
            let history = agnes(condensed, FlexibleBetaLinkage::new(-0.25)).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn agnes_ward_regression() {
        test_clustering_condensed("AGNES", "ward", SquaredEuclidean, |condensed, min_clusters| {
            let history = agnes(condensed, WardLinkage).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn agnes_minimum_variance_increase_regression() {
        test_clustering_condensed("AGNES", "mivar", Euclidean, |condensed, min_clusters| {
            let history = agnes(condensed, MinimumVarianceIncreaseLinkage).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn agnes_minimum_sum_squares_regression() {
        test_clustering_condensed("AGNES", "mnssq", Euclidean, |condensed, min_clusters| {
            let history = agnes(condensed, MinimumSumSquaresLinkage).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }

    #[test]
    fn agnes_minimum_variance_regression() {
        test_clustering_condensed("AGNES", "mnvar", Euclidean, |condensed, min_clusters| {
            let history = agnes(condensed, MinimumVarianceLinkage).unwrap();
            {
                let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                (labels, history.last().unwrap().distance)
            }
        });
    }
}
