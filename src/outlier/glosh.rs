/// GLOSH - Global-Local Outlier Scores from Hierarchies.
///
/// Reference: Campello, Moulavi, Zimek, Sander - *Hierarchical Density Estimates for
/// Data Clustering, Visualization, and Outlier Detection*, ACM TKDD 10(1), 2015.
///
/// GLOSH scores are a by-product of HDBSCAN* extraction.  The score of a point
/// reflects how far its core distance is from the densest region in its cluster:
///   score = 1 - dmin / core_dist
/// where `dmin` is the minimum merge distance in the subtree the point belongs to.
/// Scores are in [0, 1]; higher means more outlier-like.
use crate::Float;
use crate::cluster::hdbscan::HdbscanHierarchy;
use crate::cluster::hdbscan::extraction::extract_hdbscan_hierarchy_hdbscan;
use crate::outlier::common::{OutlierResult, make_outlier_result};

/// Compute GLOSH outlier scores from an HDBSCAN* hierarchy.
///
/// `min_cluster_size` controls the minimum cluster size used during extraction
/// (same parameter as in HDBSCAN* itself, typically >= 2).
///
/// Returns scores in [0, 1].  Higher score indicates a more outlier-like point.
pub fn glosh<F: Float>(
    hierarchy: &HdbscanHierarchy<F>, min_cluster_size: usize,
) -> OutlierResult<F> {
    let result = extract_hdbscan_hierarchy_hdbscan(hierarchy, min_cluster_size, false);
    make_outlier_result(result.glosh, "GLOSH", false, F::zero(), F::zero(), F::one())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::hdbscan::extraction::extract_hdbscan_hierarchy_hdbscan;
    use crate::cluster::hdbscan::slink_hdbscan::slink_hdbscan;

    fn make_two_cluster_data() -> Vec<Vec<f64>> {
        // Two tight clusters plus an obvious isolated outlier
        let mut pts = Vec::new();
        for i in 0..8 {
            pts.push(vec![i as f64 * 0.1, 0.0]); // cluster A: 0.0 - 0.7
        }
        for i in 0..8 {
            pts.push(vec![10.0 + i as f64 * 0.1, 0.0]); // cluster B: 10.0 - 10.7
        }
        pts.push(vec![100.0, 0.0]); // outlier far from both clusters
        pts
    }

    #[test]
    fn outlier_scores_higher_than_inliers() {
        use crate::TableWithDistance;
        use crate::distance::Euclidean;

        let pts = make_two_cluster_data();
        let data: TableWithDistance<f64, Vec<f64>, Euclidean, f64> =
            TableWithDistance::with_distance(&pts, Euclidean);
        let hierarchy = slink_hdbscan(&data, 2).unwrap();
        let result = glosh(&hierarchy, 2);

        let n = pts.len();
        let outlier_idx = n - 1; // the isolated point at 100.0

        // The outlier should have a higher GLOSH score than all cluster A members
        let cluster_a_max = (0..8).map(|i| result.scores[i]).fold(0.0f64, f64::max);
        assert!(
            result.scores[outlier_idx] > cluster_a_max,
            "outlier score {} should exceed cluster A max {}",
            result.scores[outlier_idx],
            cluster_a_max
        );
    }

    #[test]
    fn scores_are_in_unit_interval() {
        use crate::TableWithDistance;
        use crate::distance::Euclidean;

        let pts = make_two_cluster_data();
        let data: TableWithDistance<f64, Vec<f64>, Euclidean, f64> =
            TableWithDistance::with_distance(&pts, Euclidean);
        let hierarchy = slink_hdbscan(&data, 2).unwrap();
        let result = glosh(&hierarchy, 2);

        for &s in &result.scores {
            assert!((0.0..=1.0).contains(&s), "score {s} out of [0, 1]");
        }
    }

    #[test]
    fn matches_extraction_glosh_directly() {
        use crate::TableWithDistance;
        use crate::distance::Euclidean;

        let pts = make_two_cluster_data();
        let data: TableWithDistance<f64, Vec<f64>, Euclidean, f64> =
            TableWithDistance::with_distance(&pts, Euclidean);
        let hierarchy = slink_hdbscan(&data, 2).unwrap();
        let result = glosh(&hierarchy, 2);
        let direct = extract_hdbscan_hierarchy_hdbscan(&hierarchy, 2, false);
        assert_eq!(result.scores, direct.glosh);
    }
}
