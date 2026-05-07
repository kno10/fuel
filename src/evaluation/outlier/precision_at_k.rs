use crate::evaluation::outlier::{score_equal, sort_score_label};

fn sorted_groups<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> Vec<(usize, usize, f64)> {
    let pairs = sort_score_label(scores, labels);

    let mut groups = Vec::new();
    let mut i = 0;
    while i < pairs.len() {
        let score = pairs[i].0;
        let mut gp = 0usize;
        let mut gs = 0usize;
        while i < pairs.len() && score_equal(pairs[i].0, score) {
            if pairs[i].1 == 1 {
                gp += 1;
            }
            gs += 1;
            i += 1;
        }
        groups.push((gp, gs, score));
    }
    groups
}

/// Precision at k (tie-aware with average tie orderings).
pub fn precision_at_k<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L], k: usize,
) -> f64 {
    assert_eq!(scores.len(), labels.len(), "scores and labels mismatch");
    assert!(k > 0 && k <= scores.len(), "k out of bounds");

    let groups = sorted_groups(scores, labels);
    let mut cum_pos = 0f64;
    let mut cum_total = 0usize;

    for &(gp, gs, _) in &groups {
        if cum_total + gs < k {
            cum_pos += gp as f64;
            cum_total += gs;
            continue;
        }

        let within = k - cum_total;
        let fraction_pos = if gs == 0 { 0.0 } else { gp as f64 * within as f64 / gs as f64 };
        let total_at_k = k as f64;
        return (cum_pos + fraction_pos) / total_at_k;
    }

    0.0
}

/// Adjusted R-Precision.
pub fn adjusted_r_precision<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let npos = labels.iter().filter(|&&l| l.into() != 0).count() as f64;
    let n = scores.len() as f64;
    super::adjusted_value(r_precision(scores, labels), npos / n)
}

/// R-Precision: precision at k = number of positives.
pub fn r_precision<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let npos = labels.iter().filter(|&&l| if l.into() != 0 { 1 } else { 0 } == 1).count();
    if npos == 0 {
        return 0.0;
    }
    precision_at_k(scores, labels, npos)
}

#[cfg(test)]
mod tests {
    use super::{precision_at_k, r_precision};

    #[test]
    fn test_precision_at_k_ties() {
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];

        let p1 = precision_at_k(&scores, &labels, 1);
        let p2 = precision_at_k(&scores, &labels, 2);
        let p3 = precision_at_k(&scores, &labels, 3);
        let rp = r_precision(&scores, &labels);

        assert!((p1 - 1.0).abs() < 1e-12);
        assert!((p2 - 0.75).abs() < 1e-12);
        assert!((p3 - (2.0 / 3.0)).abs() < 1e-12);
        assert!((rp - (2.0 / 3.0)).abs() < 1e-12);
    }
}
