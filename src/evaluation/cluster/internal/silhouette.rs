use std::collections::BTreeMap;

use super::helpers::{NoiseHandling, SilhouetteStats, count_labels, euc, mean_std};
use crate::ParMap;

/// Simplified silhouette measure (triangle inequality approximation).
#[must_use]
pub fn simplified_silhouette(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling,
    penalize: bool,
) -> SilhouetteStats<f64> {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 {
        return SilhouetteStats { mean: 0.0, stddev: 0.0, values: Vec::new() };
    }

    let is_noise = |l: isize| noise_label == Some(l);
    let label_count = count_labels(labels);
    let ignored = match nh {
        NoiseHandling::IgnoreNoise => {
            noise_label.and_then(|nl| label_count.get(&nl)).copied().unwrap_or(0)
        }
        _ => 0,
    };

    // Accumulate per-label centroid sums in one pass.
    // Noise clusters never get a centroid (matching original cluster_centroids semantics).
    // Under IgnoreNoise: also skip non-noise singletons.
    // Under TreatNoiseAsSingletons/MergeNoise: include non-noise singletons
    //   (centroid of a singleton = the point itself; euc to centroid = euc to member).
    let dims = data[0].len();
    let mut centroid_sum: BTreeMap<isize, Vec<f64>> = BTreeMap::new();
    for (i, &l) in labels.iter().enumerate() {
        if is_noise(l) {
            continue;
        }
        if nh == NoiseHandling::IgnoreNoise && label_count.get(&l).copied().unwrap_or(0) <= 1 {
            continue;
        }
        let sum = centroid_sum.entry(l).or_insert_with(|| vec![0.0; dims]);
        for (c, &x) in sum.iter_mut().zip(&data[i]) {
            *c += x;
        }
    }
    let centroids: BTreeMap<isize, Vec<f64>> = centroid_sum
        .into_iter()
        .map(|(l, mut sum)| {
            let cnt = label_count.get(&l).copied().unwrap_or(1) as f64;
            for c in &mut sum {
                *c /= cnt;
            }
            (l, sum)
        })
        .collect();

    // Noise points used as individual b candidates (min distance, not mean).
    // Under IgnoreNoise: noise is excluded entirely.
    // Under TreatNoiseAsSingletons: each is its own singleton — direct distance.
    // Under MergeNoise: the merged noise cluster has no centroid — direct distance per member.
    let noise_pts: Vec<usize> = if nh == NoiseHandling::IgnoreNoise {
        Vec::new()
    } else {
        (0..n).filter(|&i| is_noise(labels[i])).collect()
    };

    let point_results: Vec<(f64, bool)> = (0..n).par_map(|p| {
        let my_label = labels[p];
        if is_noise(my_label) && nh != NoiseHandling::MergeNoise {
            return (0.0, false);
        }
        let my_count = label_count.get(&my_label).copied().unwrap_or(0);
        if my_count <= 1 {
            return (0.0, false);
        }
        let center = match centroids.get(&my_label) {
            Some(c) => c,
            None => return (0.0, false),
        };
        let a = euc(center, &data[p]);

        let mut b = f64::INFINITY;
        for (&l, cj) in &centroids {
            if l != my_label {
                b = b.min(euc(cj, &data[p]));
            }
        }
        for &q in &noise_pts {
            b = b.min(euc(&data[p], &data[q]));
        }

        let s = if !b.is_finite() || (a == 0.0 && b == 0.0) { 0.0 } else { (b - a) / a.max(b) };
        (s, true)
    });

    let values: Vec<f64> = point_results.iter().map(|(v, _)| *v).collect();
    let included: Vec<f64> = point_results
        .into_iter()
        .filter_map(|(v, include)| if include { Some(v) } else { None })
        .collect();

    let mut penalty = 1.0;
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        penalty = (n - ignored) as f64 / n as f64;
    }
    let (mean, stddev) = mean_std(&included);
    SilhouetteStats { mean: penalty * mean, stddev: penalty * stddev, values }
}

/// Full silhouette score.
#[must_use]
pub fn silhouette(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling,
    penalize: bool,
) -> SilhouetteStats<f64> {
    assert_eq!(data.len(), labels.len());
    let n = data.len();
    if n == 0 {
        return SilhouetteStats { mean: 0.0, stddev: 0.0, values: Vec::new() };
    }

    let is_noise = |l: isize| noise_label == Some(l);
    let label_count = count_labels(labels);
    let ignored = match nh {
        NoiseHandling::IgnoreNoise => {
            noise_label.and_then(|nl| label_count.get(&nl)).copied().unwrap_or(0)
        }
        _ => 0,
    };

    // Two noise points with the same label are NOT in the same cluster under
    // TreatNoiseAsSingletons (each noise point is its own singleton).
    let same_cluster = |la: isize, lb: isize| {
        la == lb && !(is_noise(la) && nh == NoiseHandling::TreatNoiseAsSingletons)
    };

    let point_results: Vec<(f64, bool)> = (0..n).par_map(|p| {
        let my_label = labels[p];
        if is_noise(my_label) && nh != NoiseHandling::MergeNoise {
            return (0.0, false);
        }
        let my_count = label_count.get(&my_label).copied().unwrap_or(0);
        if my_count <= 1 {
            return (0.0, false);
        }

        let mut same_sum = 0.0f64;
        // Per-label sum of distances to other-cluster points; used to compute means.
        let mut other_sums: BTreeMap<isize, f64> = BTreeMap::new();
        // Under TreatNoiseAsSingletons: each noise point is its own singleton cluster;
        // b candidate = direct distance (not a cluster mean).
        let mut b_noise_singleton = f64::INFINITY;

        for q in 0..n {
            if q == p {
                continue;
            }
            let lq = labels[q];
            if is_noise(lq) && nh == NoiseHandling::IgnoreNoise {
                continue;
            }
            let d = euc(&data[p], &data[q]);
            if same_cluster(my_label, lq) {
                same_sum += d;
            } else if is_noise(lq) && nh == NoiseHandling::TreatNoiseAsSingletons {
                b_noise_singleton = b_noise_singleton.min(d);
            } else {
                *other_sums.entry(lq).or_insert(0.0) += d;
            }
        }

        let a = same_sum / (my_count as f64 - 1.0);

        let mut b = b_noise_singleton;
        for (&l, &sum) in &other_sums {
            let cnt = label_count.get(&l).copied().unwrap_or(0);
            // Under IgnoreNoise: skip non-noise singleton clusters (matching original).
            if nh == NoiseHandling::IgnoreNoise && cnt <= 1 {
                continue;
            }
            if cnt > 0 {
                b = b.min(sum / cnt as f64);
            }
        }

        let s = if b.is_finite() && (a > 0.0 || b > 0.0) { (b - a) / a.max(b) } else { 0.0 };
        (s, true)
    });

    let values: Vec<f64> = point_results.iter().map(|(v, _)| *v).collect();
    let included: Vec<f64> = point_results
        .into_iter()
        .filter_map(|(v, include)| if include { Some(v) } else { None })
        .collect();

    let mut penalty = 1.0;
    if penalize && nh == NoiseHandling::IgnoreNoise && ignored > 0 {
        penalty = (n - ignored) as f64 / n as f64;
    }
    let (mean, stddev) = mean_std(&included);
    SilhouetteStats { mean: penalty * mean, stddev: penalty * stddev, values }
}
