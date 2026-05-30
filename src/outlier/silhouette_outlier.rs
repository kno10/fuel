/// Silhouette-based outlier detection.
///
/// Reference: Rousseeuw - *Silhouettes: A graphical aid to the interpretation
/// and validation of cluster analysis*, JCAM 20, 1987.
///
/// Low silhouette values (close to -1) indicate poorly assigned points that
/// are more similar to a neighbouring cluster than their own.  High values
/// (close to 1) indicate well-assigned inliers.
///
/// **Note**: this function is `f64`-only because the underlying `silhouette`
/// evaluation code is not generic.
use crate::evaluation::cluster::internal::helpers::NoiseHandling;
use crate::evaluation::cluster::internal::silhouette::silhouette;
use crate::outlier::common::{OutlierResult, make_outlier_result};

/// Silhouette-based outlier detection.
///
/// `data`         - points as `&[Vec<f64>]`.
/// `labels`       - cluster label per point; use the `NOISE` sentinel (-1) for
///                  unclustered points.
/// `noise_label`  - sentinel value for noise/unclustered points
///                  (pass `Some(NOISE)` when using DBSCAN-style labels,
///                  `None` when every point is in a cluster).
///
/// Noise points are treated as singletons.  Points with a silhouette of 1 are
/// perfect inliers; points with a silhouette close to -1 are likely outliers.
///
/// Returns scores with `ascending=true` (lower score = more outlier-like),
/// `baseline=0.5`, theoretical range [-1, 1].
pub fn silhouette_outlier(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>,
) -> OutlierResult<f64> {
    assert_eq!(data.len(), labels.len(), "data and labels must have the same length");

    let stats = silhouette(data, labels, noise_label, NoiseHandling::TreatNoiseAsSingletons, false);

    make_outlier_result(
        stats.values,
        "SilhouetteOutlier",
        true, // ascending: low silhouette = outlier
        0.5,
        -1.0,
        1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::dbscan::NOISE; // for NOISE sentinel in test labels

    #[test]
    fn well_clustered_point_scores_higher_than_border() {
        // Three perfect clusters of 3 points each on a line:
        // cluster 0: 0.0 0.1 0.2   cluster 1: 5.0 5.1 5.2   cluster 2: 10.0 10.1 10.2
        let data: Vec<Vec<f64>> = vec![
            vec![0.0],
            vec![0.1],
            vec![0.2],
            vec![5.0],
            vec![5.1],
            vec![5.2],
            vec![10.0],
            vec![10.1],
            vec![10.2],
        ];
        let labels: Vec<isize> = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let scores = silhouette_outlier(&data, &labels, None);

        // Points in compact, well-separated clusters should score highly
        assert!(scores.scores[0] > 0.0, "scores[0] = {}", scores.scores[0]);
        assert!(scores.scores[4] > 0.0, "scores[4] = {}", scores.scores[4]);
    }

    #[test]
    fn noise_point_treated_as_singleton_scores_lower() {
        // Same clusters but point 3 is declared noise; as a singleton its
        // silhouette is defined as 0 by TreatNoiseAsSingletons.
        let data: Vec<Vec<f64>> =
            vec![vec![0.0], vec![0.1], vec![0.2], vec![5.0], vec![10.0], vec![10.1], vec![10.2]];
        let labels: Vec<isize> = vec![0, 0, 0, NOISE, 1, 1, 1];
        let scores = silhouette_outlier(&data, &labels, Some(NOISE));

        // Inlier cluster members score positively
        assert!(scores.scores[0] > 0.0);
        // Noise singleton defined as 0, which is below typical inlier score
        assert!(scores.scores[3] < scores.scores[0]);
    }
}
