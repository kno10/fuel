use super::helpers::{NoiseHandling, cluster_centroids, group_by_label, sq_euc};

/// Variance ratio criterion (Calinski-Harabasz index).
#[must_use]
pub fn variance_ratio_criterion(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling,
    penalize: bool,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    let (clusters, ignored) = group_by_label(labels, noise_label, nh);
    if clusters.len() <= 1 {
        return 0.0;
    }
    let centroids = cluster_centroids(data, &clusters, nh);

    let dim = data[0].len();
    let mut overall = vec![0.0; dim];
    let mut overall_count = 0usize;
    let mut clustercount = 0usize;

    for (i, (members, is_noise)) in clusters.iter().enumerate() {
        if members.len() <= 1 || *is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => {
                    clustercount += members.len();
                    for &p in members {
                        for (d, val) in data[p].iter().enumerate() {
                            overall[d] += *val;
                        }
                    }
                    overall_count += members.len();
                    continue;
                }
                NoiseHandling::MergeNoise => {}
            }
        }
        if let Some(c) = &centroids[i] {
            clustercount += 1;
            for d in 0..dim {
                overall[d] += c[d] * members.len() as f64;
            }
            overall_count += members.len();
        }
    }

    if overall_count == 0 || clustercount <= 1 {
        return 0.0;
    }
    for x in &mut overall {
        *x /= overall_count as f64;
    }

    let mut a = 0.0;
    let mut b = 0.0;

    for (i, (members, is_noise)) in clusters.iter().enumerate() {
        if members.len() <= 1 || *is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => {
                    for &p in members {
                        b += sq_euc(&overall, &data[p]);
                    }
                    continue;
                }
                NoiseHandling::MergeNoise => {}
            }
        }
        let c = centroids[i].as_ref().expect("centroid required");
        for &p in members {
            a += sq_euc(c, &data[p]);
            b += sq_euc(&overall, &data[p]);
        }
    }

    if a == 0.0 {
        return 0.0;
    }
    let mut vrc = ((b - a) / a) * ((n as f64 - clustercount as f64) / (clustercount as f64 - 1.0));
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        vrc *= (n - ignored) as f64 / n as f64;
    }
    vrc
}
