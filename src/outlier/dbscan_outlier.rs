/// DBSCAN-based outlier detection.
///
/// Two functions are provided:
///
/// - [`dbscan_outlier`]: three-tier scoring (core=0, border=0.5, noise=1.0).
///   Requires a [`DbscanResult`] from [`dbscan`].
///
/// - [`noise_as_outliers`]: binary scoring (noise=1, assigned=0) applicable to
///   any label vector using the NOISE sentinel (-1).
///
/// Reference: He, Xu, Deng - *Discovering cluster-based local outliers*,
/// Pattern Recognition Letters 24(9-10), 2003 (DBSCAN outlier variant).
///
/// [`dbscan`]: crate::cluster::dbscan::dbscan
/// [`DbscanResult`]: crate::cluster::dbscan::DbscanResult
use crate::Float;
use crate::ParMap;
use crate::cluster::dbscan::{DbscanResult, NOISE};
use crate::outlier::common::OutlierResult;

/// DBSCAN-based outlier detection with core/border/noise distinction.
///
/// - Core points (|Neps| >= min_points): score **0.0** (clear inliers).
/// - Border points (in a cluster, but not core): score **0.5** (local outliers).
/// - Noise points (label == NOISE): score **1.0** (global outliers).
///
/// Returns scores in [0, 1] with `ascending=false` (higher = more outlier-like).
pub fn dbscan_outlier<F: Float>(result: &DbscanResult) -> OutlierResult<F> {
    let scores: Vec<F> = (0..result.labels.len()).par_map(|i| {
        let label = result.labels[i];
        let is_core = result.core[i];
        if label == NOISE {
            F::one()
        } else if is_core {
            F::zero()
        } else {
            F::from_f64(0.5).unwrap()
        }
    });

    OutlierResult::builder(scores, "DBSCANOutlier")
        .theoretical_min(F::zero())
        .theoretical_max(F::one())
        .build()
}

/// Generic binary outlier scoring: noise points score 1, all others score 0.
///
/// Any label value of `NOISE` (-1) is treated as an outlier.  This is
/// applicable to any clustering algorithm that uses the `-1` sentinel for
/// noise (DBSCAN, HDBSCAN extraction, etc.).
pub fn noise_as_outliers<F: Float>(labels: &[isize]) -> OutlierResult<F> {
    let scores: Vec<F> =
        (0..labels.len()).par_map(|i| if labels[i] == NOISE { F::one() } else { F::zero() });
    OutlierResult::builder(scores, "NoiseAsOutliers")
        .theoretical_min(F::zero())
        .theoretical_max(F::one())
        .build()
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::cluster::dbscan::{NOISE, dbscan};
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    fn make_tree(
        points: &[Vec<f64>],
    ) -> (TableWithDistance<'_, f64, Vec<f64>, Euclidean, f64>, VPTree<f64>) {
        let data = TableWithDistance::with_distance(points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree = VPTree::new(&data, 2, &mut rng);
        (data, tree)
    }

    fn two_cluster_points() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1], // cluster 0
            vec![5.0, 5.0],
            vec![5.1, 5.0],
            vec![5.0, 5.1],  // cluster 1
            vec![10.0, 0.0], // noise
        ]
    }

    #[test]
    fn noise_point_scores_one_border_half_core_zero() {
        let points = two_cluster_points();
        let (data, tree) = make_tree(&points);
        let result = dbscan(&tree, &data, 0.2, 3).unwrap();
        let scores = dbscan_outlier::<f64>(&result);

        // The isolated point at index 6 must be noise
        assert_eq!(result.labels[6], NOISE);
        assert_eq!(scores.scores[6], 1.0);

        // Core points in each cluster score 0
        let core_count_cluster0 = (0..3).filter(|&i| result.core[i]).count();
        assert!(core_count_cluster0 > 0);
        for i in 0..3 {
            if result.core[i] {
                assert_eq!(scores.scores[i], 0.0);
            }
        }
    }

    #[test]
    fn noise_as_outliers_matches_noise_label() {
        let points = two_cluster_points();
        let (data, tree) = make_tree(&points);
        let result = dbscan(&tree, &data, 0.2, 3).unwrap();
        let scores = noise_as_outliers::<f64>(&result.labels);

        // Noise point scores 1, all others 0
        assert_eq!(scores.scores[6], 1.0);
        for i in 0..6 {
            assert_eq!(scores.scores[i], 0.0);
        }
    }
}
