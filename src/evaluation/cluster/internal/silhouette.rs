use super::helpers::{
    NoiseHandling, SilhouetteStats, build_clusters, cluster_centroids, euc, mean_std,
};

/// Simplified silhouette measure (triangle inequality approximation).
#[must_use]
pub fn simplified_silhouette(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
    penalize: bool,
) -> SilhouetteStats<f64> {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 {
        return SilhouetteStats {
            mean: 0.0,
            stddev: 0.0,
            values: Vec::new(),
        };
    }

    let (clusters, ignored) = build_clusters(labels, noise_label, nh);
    let centroids = cluster_centroids(data, &clusters, nh);
    let mut values = vec![0.0; n];
    let mut included = Vec::new();

    for (i, cluster) in clusters.iter().enumerate() {
        if cluster.members.len() <= 1 {
            continue;
        }
        if cluster.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }

        let center = centroids[i].as_ref().expect("centroid required");
        for &p in &cluster.members {
            let a = euc(center, &data[p]);
            let mut b = f64::INFINITY;

            for (j, other) in clusters.iter().enumerate() {
                if i == j {
                    continue;
                }
                match &centroids[j] {
                    Some(cj) => {
                        b = b.min(euc(cj, &data[p]));
                    }
                    None => {
                        if nh == NoiseHandling::IgnoreNoise {
                            continue;
                        }
                        for &q in &other.members {
                            b = b.min(euc(&data[p], &data[q]));
                        }
                    }
                }
            }

            if !b.is_finite() {
                b = a;
            }
            let s = if a == 0.0 && b == 0.0 {
                0.0
            } else {
                (b - a) / a.max(b)
            };
            values[p] = s;
            included.push(s);
        }
    }

    let mut penalty = 1.0;
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        penalty = (n - ignored) as f64 / n as f64;
    }
    let (mean, stddev) = mean_std(&included);
    SilhouetteStats {
        mean: penalty * mean,
        stddev: penalty * stddev,
        values,
    }
}

/// Full silhouette score.
#[must_use]
pub fn silhouette(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
    penalize: bool,
) -> SilhouetteStats<f64> {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 {
        return SilhouetteStats {
            mean: 0.0,
            stddev: 0.0,
            values: Vec::new(),
        };
    }

    let (clusters, ignored) = build_clusters(labels, noise_label, nh);
    let mut values = vec![0.0; n];
    let mut included = Vec::new();

    if clusters.len() <= 1 {
        return SilhouetteStats {
            mean: 0.0,
            stddev: 0.0,
            values,
        };
    }

    for (i, cluster) in clusters.iter().enumerate() {
        if cluster.members.len() <= 1 || cluster.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }

        for &p in &cluster.members {
            let a = if cluster.members.len() <= 1 {
                0.0
            } else {
                let mut sum = 0.0;
                for &q in &cluster.members {
                    if p != q {
                        sum += euc(&data[p], &data[q]);
                    }
                }
                sum / (cluster.members.len() as f64 - 1.0)
            };

            let mut b = f64::INFINITY;
            for (j, other) in clusters.iter().enumerate() {
                if i == j {
                    continue;
                }
                if (other.members.len() <= 1 || other.is_noise) && nh == NoiseHandling::IgnoreNoise
                {
                    continue;
                }
                if (other.members.len() <= 1 || other.is_noise)
                    && nh == NoiseHandling::TreatNoiseAsSingletons
                {
                    for &q in &other.members {
                        b = b.min(euc(&data[p], &data[q]));
                    }
                    continue;
                }
                let avg = other
                    .members
                    .iter()
                    .map(|&q| euc(&data[p], &data[q]))
                    .sum::<f64>()
                    / other.members.len() as f64;
                b = b.min(avg);
            }

            let s = if b.is_finite() && (a > 0.0 || b > 0.0) {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            values[p] = s;
            included.push(s);
        }
    }

    let mut penalty = 1.0;
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        penalty = (n - ignored) as f64 / n as f64;
    }
    let (mean, stddev) = mean_std(&included);
    SilhouetteStats {
        mean: penalty * mean,
        stddev: penalty * stddev,
        values,
    }
}
