//! Geometric nearest-neighbor chain clustering using stored-data centroids.
//!
//! This variant is a stored-data approach for vector data and squared Euclidean
//! linkage functions.  It avoids the full condensed distance matrix by
//! maintaining only cluster centres and merge heights.
//!
//! Because it relies on centroid merging and the König-Huygens identity, it is
//! only suitable for geometric linkages such as `CentroidLinkage`,
//! `GroupAverageLinkage`, `WardLinkage`, and similar centroid-based methods.

use crate::cluster::hierarchical::common::{find_active, shrink_active_end};
use crate::cluster::hierarchical::{Builder, GeometricLinkage, MergeHistory, idsize};
use crate::{DistanceData, Float};

/// Perform NN-Chain clustering on vector data with geometric linkage,
/// also known as stored-data approach in literature.
///
/// This requires the data to be vectors, and the distance function to be squared Euclidean.
/// In other cases, the results may be undesirable or not match expectations.
/// For example, group average linkage with Euclidean will find the squared Euclidean partitions,
/// because we internally map Euclidean to squared Euclidean and only restore the distance scale for output.
///
/// This variant avoids the `O(n^2)` condensed matrix and instead keeps only
/// current cluster representatives. Runtime is still `O(n^3)` in the worst
/// case, but memory usage is `O(n * dim)`.
#[must_use]
pub fn geometric_nn_chain<F: Float, L: GeometricLinkage<F> + Copy, D>(
    data: &D, linkage: L,
) -> MergeHistory<F>
where
    D: crate::VectorData<F> + DistanceData<F>,
{
    let n = data.nrows();
    assert!(n > 0, "number of points must be positive");

    let _dim = data.dims();
    let squared = data.is_squared_distance();

    let mut builder = Builder::<F>::new(n);
    let mut clustermap: Vec<idsize> = (0..(n as idsize)).collect();
    let mut clusters: Vec<Option<Vec<F>>> = (0..n).map(|i| Some(data.point(i).to_vec())).collect();
    let mut heights = vec![F::zero(); n];
    let mut end = n;
    let mut chain: Vec<usize> = Vec::with_capacity(n.min(64));
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
                chain.truncate(chain.len() - 2);
                continue;
            }
            chain.pop();
        }

        let mut min_dist = {
            let aa = clusters[a].as_ref().expect("a center must exist");
            let bb = clusters[b].as_ref().expect("b center must exist");
            let size_a = builder.get_size(clustermap[a]);
            let size_b = builder.get_size(clustermap[b]);
            linkage.linkage(aa, size_a, bb, size_b, heights[a], heights[b])
        };

        loop {
            let mut c = b;
            let a_center = clusters[a].as_ref().expect("a center must exist");
            let size_a = builder.get_size(clustermap[a]);
            for i in 0..end {
                if i == a || i == b || clustermap[i] == idsize::MAX {
                    continue;
                }
                let i_center = clusters[i].as_ref().expect("i center must exist");
                let size_i = builder.get_size(clustermap[i]);
                let d = linkage.linkage(a_center, size_a, i_center, size_i, heights[a], heights[i]);
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

        let merged_center = {
            let cx = clusters[x].as_ref().expect("x center must exist");
            let cy = clusters[y].as_ref().expect("y center must exist");
            linkage.merge(cx, size_x, cy, size_y, heights[x], heights[y])
        };

        let merged_height = linkage.merge_height(
            clusters[y].as_ref().expect("y center must exist"),
            size_y,
            clusters[x].as_ref().expect("x center must exist"),
            size_x,
            heights[y],
            heights[x],
        );

        clustermap[y] = new_id;
        clustermap[x] = idsize::MAX;
        clusters[y] = Some(merged_center);
        heights[y] = merged_height;
        heights[x] = F::nan();
        clusters[x] = None;

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
    use super::geometric_nn_chain;
    use crate::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::test::test_clustering_table;
    use crate::cluster::hierarchical::{
        CentroidLinkage, GroupAverageLinkage, MedianLinkage, MinimumSumSquaresLinkage,
        MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage, WardLinkage, nn_chain,
    };
    use crate::distance::SquaredEuclidean;
    use crate::{CondensedDistanceMatrix, TableWithDistance};

    #[test]
    fn geometric_nn_chain_average_regression() {
        // Only SquaredEuclidean is Geometric
        test_clustering_table(
            "GeometricNNChain",
            "average",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = geometric_nn_chain(access, GroupAverageLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn geometric_nn_chain_centroid_regression() {
        test_clustering_table(
            "GeometricNNChain",
            "centroid",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = geometric_nn_chain(access, CentroidLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn geometric_nn_chain_median_regression() {
        test_clustering_table(
            "GeometricNNChain",
            "median",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = geometric_nn_chain(access, MedianLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn geometric_nn_chain_ward_regression() {
        test_clustering_table(
            "GeometricNNChain",
            "ward",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = geometric_nn_chain(access, WardLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn geometric_nn_chain_minimum_sum_squares_regression() {
        test_clustering_table(
            "GeometricNNChain",
            "mnssq",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = geometric_nn_chain(access, MinimumSumSquaresLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn geometric_nn_chain_minimum_variance_regression() {
        test_clustering_table(
            "GeometricNNChain",
            "mnvar",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = geometric_nn_chain(access, MinimumVarianceLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn geometric_nn_chain_minimum_variance_increase_regression() {
        test_clustering_table(
            "GeometricNNChain",
            "mivar",
            SquaredEuclidean,
            |access, min_clusters| {
                let history = geometric_nn_chain(access, MinimumVarianceIncreaseLinkage);
                {
                    let labels = cut_dendrogram_by_number_of_clusters(&history, min_clusters);
                    (labels, history.last().unwrap().distance)
                }
            },
        );
    }

    #[test]
    fn geometric_nn_chain_average_matches_nn_chain_distances() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let data: TableWithDistance<'_, f64, Vec<f64>, SquaredEuclidean, f64> =
            TableWithDistance::with_distance(&points, SquaredEuclidean);
        let condensed: CondensedDistanceMatrix<f64> = CondensedDistanceMatrix::new_from_data(&data);

        let geometric_history = geometric_nn_chain(&data, GroupAverageLinkage);
        let reference_history = nn_chain(&condensed, GroupAverageLinkage);

        assert_eq!(geometric_history.len(), reference_history.len());
        for (geo_merge, ref_merge) in geometric_history.iter().zip(reference_history.iter()) {
            assert_eq!(geo_merge.idx1, ref_merge.idx1);
            assert_eq!(geo_merge.idx2, ref_merge.idx2);
            let diff = (geo_merge.distance as f64) - (ref_merge.distance as f64);
            assert!(diff.abs() < 1e-12);
            assert_eq!(geo_merge.size, ref_merge.size);
        }
    }
}
