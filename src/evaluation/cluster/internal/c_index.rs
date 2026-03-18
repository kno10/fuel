use super::helpers::{NoiseHandling, build_clusters, euc};
use std::cmp::Ordering;

/// C-index measure for clustering compactness.
#[must_use]
pub fn c_index(
    data: &[Vec<f64>],
    labels: &[isize],
    noise_label: Option<isize>,
    nh: NoiseHandling,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    let (clusters, _) = build_clusters(labels, noise_label, nh);

    let mut w = 0usize;
    let mut theta = 0.0;
    for cl in &clusters {
        if (cl.members.len() <= 1 || cl.is_noise) && nh == NoiseHandling::IgnoreNoise {
            continue;
        }
        w += cl.members.len() * cl.members.len().saturating_sub(1) / 2;
        for i in 0..cl.members.len() {
            for j in (i + 1)..cl.members.len() {
                theta += euc(&data[cl.members[i]], &data[cl.members[j]]);
            }
        }
    }
    if w == 0 {
        return 1.0;
    }

    let mut considered = Vec::new();
    for cl in &clusters {
        if (cl.members.len() <= 1 || cl.is_noise) && nh == NoiseHandling::IgnoreNoise {
            continue;
        }
        considered.extend_from_slice(&cl.members);
    }
    considered.sort_unstable();
    considered.dedup();

    let mut dists = Vec::new();
    for i in 0..considered.len() {
        for j in (i + 1)..considered.len() {
            dists.push(euc(&data[considered[i]], &data[considered[j]]));
        }
    }
    if dists.is_empty() {
        return 1.0;
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let min = dists.iter().take(w).sum::<f64>();
    let max = dists.iter().rev().take(w).sum::<f64>();
    if max > min {
        (theta - min) / (max - min)
    } else {
        1.0
    }
}
