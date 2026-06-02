#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::if_not_else)]

use super::helpers::{NoiseHandling, cluster_centroids, euc, group_by_label};

/// Davies–Bouldin index for clustering quality.
#[must_use]
pub fn davies_bouldin_index(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling, p: f64,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    let (clusters, _) = group_by_label(labels, noise_label, nh);
    if clusters.is_empty() {
        return 0.0;
    }
    let centroids = cluster_centroids(data, &clusters, nh);

    let mut within = vec![0.0; clusters.len()];
    for (i, (members, _)) in clusters.iter().enumerate() {
        if let Some(c) = &centroids[i] {
            let mut w = 0.0;
            for &m in members {
                let dist = euc(c, &data[m]);
                w += if p != 1.0 { dist.powf(p) } else { dist };
            }
            w /= members.len() as f64;
            within[i] = if p != 1.0 { w.powf(1.0 / p) } else { w };
        }
    }

    let mut vals = Vec::new();
    for i in 0..clusters.len() {
        let mut maxd = 0.0f64;
        for j in 0..clusters.len() {
            if i == j {
                continue;
            }
            match (&centroids[i], &centroids[j]) {
                (Some(ci), Some(cj)) => {
                    let bd = euc(ci, cj);
                    if bd > 0.0 {
                        maxd = maxd.max((within[i] + within[j]) / bd);
                    }
                }
                (Some(ci), None) if nh != NoiseHandling::IgnoreNoise => {
                    let mut d = f64::INFINITY;
                    for &m in &clusters[j].0 {
                        d = d.min(euc(ci, &data[m]));
                    }
                    if d.is_finite() && d > 0.0 {
                        maxd = maxd.max(within[i] / d);
                    }
                }
                (None, Some(cj)) if nh != NoiseHandling::IgnoreNoise => {
                    let mut d = f64::INFINITY;
                    for &m in &clusters[i].0 {
                        d = d.min(euc(&data[m], cj));
                    }
                    if d.is_finite() && d > 0.0 {
                        maxd = maxd.max(within[j] / d);
                    }
                }
                _ => {}
            }
        }
        vals.push(maxd);
    }

    if vals.len() > 1 { vals.iter().sum::<f64>() / vals.len() as f64 } else { 2.0 }
}
