use super::helpers::{NoiseHandling, build_clusters, cluster_centroids, euc};

/// Pakhira-Bandyopadhyay-Maulik (PBM) index.
#[must_use]
pub fn pbm_index(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    if data.is_empty() {
        return 0.0;
    }
    let (clusters, _) = build_clusters(labels, noise_label, nh);
    if clusters.is_empty() {
        return 0.0;
    }
    let centroids = cluster_centroids(data, &clusters, nh);

    let dim = data[0].len();
    let mut overall = vec![0.0; dim];
    for p in data {
        for d in 0..dim {
            overall[d] += p[d];
        }
    }
    for x in &mut overall {
        *x /= data.len() as f64;
    }

    let mut maxd = 0.0f64;
    for i in 0..clusters.len() {
        if centroids[i].is_none() && nh != NoiseHandling::TreatNoiseAsSingletons {
            continue;
        }
        for j in (i + 1)..clusters.len() {
            if centroids[j].is_none() && nh != NoiseHandling::TreatNoiseAsSingletons {
                continue;
            }
            match (&centroids[i], &centroids[j]) {
                (Some(ci), Some(cj)) => maxd = maxd.max(euc(ci, cj)),
                (None, Some(cj)) => {
                    for &m in &clusters[i].members {
                        maxd = maxd.max(euc(&data[m], cj));
                    }
                }
                (Some(ci), None) => {
                    for &m in &clusters[j].members {
                        maxd = maxd.max(euc(ci, &data[m]));
                    }
                }
                (None, None) => {
                    for &a in &clusters[i].members {
                        for &b in &clusters[j].members {
                            maxd = maxd.max(euc(&data[a], &data[b]));
                        }
                    }
                }
            }
        }
    }

    let mut a = 0.0;
    let mut b = 0.0;
    let mut n_cl = clusters.len() as f64;

    for (i, cl) in clusters.iter().enumerate() {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => {
                    n_cl -= 1.0;
                    continue;
                }
                NoiseHandling::TreatNoiseAsSingletons => {
                    for &m in &cl.members {
                        b += euc(&overall, &data[m]);
                    }
                    n_cl += cl.members.len() as f64 - 1.0;
                    continue;
                }
                NoiseHandling::MergeNoise => {}
            }
        }
        if let Some(c) = &centroids[i] {
            for &m in &cl.members {
                a += euc(c, &data[m]);
                b += euc(&overall, &data[m]);
            }
        }
    }

    if a <= 0.0 || n_cl <= 0.0 {
        0.0
    } else {
        ((1.0 / n_cl) * (b / a) * maxd).powi(2)
    }
}
