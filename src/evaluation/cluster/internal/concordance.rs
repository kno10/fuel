#![allow(clippy::cast_precision_loss, clippy::match_same_arms)]

use std::cmp::Ordering;

use super::helpers::{
    ConcordanceStats, NoiseHandling, count_labels, euc, lower_bound, upper_bound,
};

/// Compute gamma and tau concordance statistics for clustering.
#[must_use]
pub fn concordant_pairs_gamma_tau(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling,
) -> ConcordanceStats<f64> {
    assert_eq!(data.len(), labels.len());
    let n = data.len();

    let is_noise = |l: isize| noise_label == Some(l);

    // Label frequencies — needed to (a) count ignored points and (b) skip singleton
    // non-noise clusters under IgnoreNoise (matching the original cluster-loop behaviour).
    let label_count = count_labels(labels);

    let ignored = match nh {
        NoiseHandling::IgnoreNoise => {
            noise_label.and_then(|nl| label_count.get(&nl)).copied().unwrap_or(0)
        }
        _ => 0,
    };

    // Under TreatNoiseAsSingletons each noise point is its own cluster of size 1,
    // so two noise points with the same label are NOT in the same cluster.
    let same_cluster = |la: isize, lb: isize| -> bool {
        la == lb && !(is_noise(la) && nh == NoiseHandling::TreatNoiseAsSingletons)
    };

    // Is a point active for the within-cluster distance collection?
    // Noise under MergeNoise forms one cluster so noise-noise pairs count.
    // Noise under IgnoreNoise/TreatNoiseAsSingletons: no within-cluster pairs.
    let within_active = |l: isize| -> bool {
        if is_noise(l) {
            nh == NoiseHandling::MergeNoise
        } else {
            // Under IgnoreNoise we also skip singleton non-noise clusters.
            nh != NoiseHandling::IgnoreNoise || label_count.get(&l).copied().unwrap_or(0) > 1
        }
    };

    // Is a point active for the between-cluster concordance computation?
    // Under IgnoreNoise: exclude noise and singleton non-noise clusters.
    // Under TreatNoiseAsSingletons / MergeNoise: all points participate.
    let between_active = |l: isize| -> bool {
        if nh == NoiseHandling::IgnoreNoise {
            !is_noise(l) && label_count.get(&l).copied().unwrap_or(0) > 1
        } else {
            true
        }
    };

    // Pass 1: collect all within-cluster distances and sort them.
    let mut within = Vec::new();
    for i in 0..n {
        if !within_active(labels[i]) {
            continue;
        }
        for j in (i + 1)..n {
            if !within_active(labels[j]) {
                continue;
            }
            if same_cluster(labels[i], labels[j]) {
                within.push(euc(&data[i], &data[j]));
            }
        }
    }
    within.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Pass 2: count concordant / discordant between-cluster pairs.
    let mut concordant_pairs = 0u64;
    let mut discordant_pairs = 0u64;
    let mut between_pairs = 0u64;

    for i in 0..n {
        if !between_active(labels[i]) {
            continue;
        }
        for j in (i + 1)..n {
            if !between_active(labels[j]) {
                continue;
            }
            if !same_cluster(labels[i], labels[j]) {
                between_pairs += 1;
                let d = euc(&data[i], &data[j]);
                let lb = lower_bound(&within, d);
                let ub = upper_bound(&within, d);
                concordant_pairs += lb as u64;
                discordant_pairs += (within.len() - ub) as u64;
            }
        }
    }

    let denom = concordant_pairs + discordant_pairs;
    let mut gamma = if denom > 0 {
        (concordant_pairs as f64 - discordant_pairs as f64) / denom as f64
    } else {
        0.0
    };

    let n_eff = data.len().saturating_sub(ignored) as u64;
    let t = (n_eff * n_eff.saturating_sub(1)) >> 1;
    let m = (t * t.saturating_sub(1)) as f64 / 2.0;
    let wd = within.len() as u64;
    let bd = between_pairs;
    let tie = ((wd * wd.saturating_sub(1)) + (bd * bd.saturating_sub(1))) as f64 / 2.0;
    let mut tau = if m > 0.0 && (m - tie) > 0.0 {
        (concordant_pairs as f64 - discordant_pairs as f64) / ((m - tie) * m).sqrt()
    } else {
        0.0
    };

    if !gamma.is_finite() || gamma < 0.0 {
        gamma = 0.0;
    }
    if !tau.is_finite() || tau < 0.0 {
        tau = 0.0;
    }

    ConcordanceStats { gamma, tau }
}
