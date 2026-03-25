use super::helpers::{BasicDistanceStats, NoiseHandling, build_clusters, centroid, euc};

/// Compute sum-of-squared errors and related statistics for clustering.
#[must_use]
pub fn squared_errors(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling,
) -> BasicDistanceStats<f64> {
    assert_eq!(data.len(), labels.len());
    if data.is_empty() {
        return BasicDistanceStats { mean: 0.0, sum_of_squares: 0.0, rmsd: 0.0 };
    }

    let (clusters, ignored) = build_clusters(labels, noise_label, nh);
    let mut sum = 0.0;
    let mut ssq = 0.0;

    for cluster in clusters {
        if cluster.members.len() <= 1 || cluster.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }
        let c = centroid(data, &cluster.members);
        for &i in &cluster.members {
            let d = euc(&c, &data[i]);
            sum += d;
            ssq += d * d;
        }
    }

    let div = (data.len() - ignored).max(1) as f64;
    BasicDistanceStats { mean: sum / div, sum_of_squares: ssq, rmsd: (ssq / div).sqrt() }
}
