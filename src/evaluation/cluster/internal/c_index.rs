use std::cmp::Ordering;

use super::helpers::{NoiseHandling, count_labels, euc};

/// C-index measure for clustering compactness.
#[must_use]
pub fn c_index(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling,
) -> f64 {
    assert_eq!(data.len(), labels.len());
    let n = data.len();

    // Under IgnoreNoise, singleton non-noise clusters are also excluded (matching the
    // original cluster-loop behaviour where size-1 clusters were skipped).
    let label_count =
        if nh == NoiseHandling::IgnoreNoise { count_labels(labels) } else { Default::default() };

    let is_noise = |l: isize| noise_label == Some(l);
    let excluded = |l: isize| -> bool {
        nh == NoiseHandling::IgnoreNoise
            && (is_noise(l) || label_count.get(&l).copied().unwrap_or(0) <= 1)
    };
    // Under TreatNoiseAsSingletons every noise point is its own cluster of size 1,
    // so two noise points with the same label are NOT in the same cluster.
    let same_cluster = |la: isize, lb: isize| -> bool {
        la == lb && !(is_noise(la) && nh == NoiseHandling::TreatNoiseAsSingletons)
    };

    let mut w = 0usize;
    let mut theta = 0.0;
    let mut dists = Vec::new();

    for i in 0..n {
        if excluded(labels[i]) {
            continue;
        }
        for j in (i + 1)..n {
            if excluded(labels[j]) {
                continue;
            }
            let d = euc(&data[i], &data[j]);
            dists.push(d);
            if same_cluster(labels[i], labels[j]) {
                w += 1;
                theta += d;
            }
        }
    }

    if w == 0 || dists.is_empty() {
        return 1.0;
    }

    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let min = dists.iter().take(w).sum::<f64>();
    let max = dists.iter().rev().take(w).sum::<f64>();
    if max > min { (theta - min) / (max - min) } else { 1.0 }
}
