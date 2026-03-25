#![allow(clippy::cast_precision_loss, clippy::match_same_arms)]

use std::cmp::Ordering;

use super::helpers::{
    ConcordanceStats, NoiseHandling, build_clusters, euc, lower_bound, upper_bound,
};

/// Compute gamma and tau concordance statistics for clustering.
#[must_use]
pub fn concordant_pairs_gamma_tau(
    data: &[Vec<f64>], labels: &[isize], noise_label: Option<isize>, nh: NoiseHandling,
) -> ConcordanceStats<f64> {
    assert_eq!(data.len(), labels.len());
    let (clusters, ignored) = build_clusters(labels, noise_label, nh);

    let mut within = Vec::new();
    for cl in &clusters {
        if cl.members.len() <= 1 || cl.is_noise {
            match nh {
                NoiseHandling::IgnoreNoise => continue,
                NoiseHandling::TreatNoiseAsSingletons => continue,
                NoiseHandling::MergeNoise => {}
            }
        }
        for i in 0..cl.members.len() {
            for j in (i + 1)..cl.members.len() {
                within.push(euc(&data[cl.members[i]], &data[cl.members[j]]));
            }
        }
    }
    within.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let mut concordant_pairs = 0u64;
    let mut discordant_pairs = 0u64;
    let mut between_pairs = 0u64;

    for i in 0..clusters.len() {
        let c1 = &clusters[i];
        if (c1.members.len() <= 1 || c1.is_noise) && nh == NoiseHandling::IgnoreNoise {
            continue;
        }
        for c2 in clusters.iter().skip(i + 1) {
            if (c2.members.len() <= 1 || c2.is_noise) && nh == NoiseHandling::IgnoreNoise {
                continue;
            }
            between_pairs += (c1.members.len() * c2.members.len()) as u64;
            for &p in &c1.members {
                for &q in &c2.members {
                    let d = euc(&data[p], &data[q]);
                    let lb = lower_bound(&within, d);
                    let ub = upper_bound(&within, d);
                    concordant_pairs += lb as u64;
                    discordant_pairs += (within.len() - ub) as u64;
                }
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
