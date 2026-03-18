#![allow(clippy::cast_precision_loss)]

use super::helpers::{NoiseHandling, RadiusStats, build_clusters, centroid, euc};

/// Compute cluster radius statistics using `f64` internally.  This helper is
/// not part of the public evaluation API and therefore does not carry a
/// generic parameter.
#[must_use]
pub fn cluster_radius(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> RadiusStats<f64> {
    assert_eq!(data.len(), labels.len());
    let (clusters, _) = build_clusters(labels, noise_label, nh);
    if clusters.is_empty() {
        return RadiusStats {
            weighted: 0.0,
            unweighted: 0.0,
        };
    }

    let mut weighted = 0.0;
    let mut unweighted = 0.0;
    let mut cnum = 0usize;

    for cl in &clusters {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons | NoiseHandling::MergeNoise => {}
            }
        }
        let c = centroid(data, &cl.members);
        let mut maxd: f64 = 0.0;
        for &m in &cl.members {
            maxd = maxd.max(euc(&c, &data[m]));
        }
        cnum += 1;
        weighted += maxd * (cl.members.len() as f64);
        unweighted += maxd;
    }

    let n = data.len().max(1) as f64;
    RadiusStats {
        weighted: weighted / n,
        unweighted: if cnum > 0 {
            unweighted / (cnum as f64)
        } else {
            0.0
        },
    }
}
