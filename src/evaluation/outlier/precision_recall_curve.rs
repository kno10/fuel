use crate::evaluation::outlier::{score_equal, sort_score_label};

pub fn pr_curve<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> Vec<(f64, f64)> {
    let pairs = sort_score_label(scores, labels);

    let n = pairs.len();
    if n == 0 {
        return vec![(0.0, 1.0)];
    }

    let npos = pairs.iter().filter(|(_, l)| *l == 1).count();
    if npos == 0 {
        // no positive: precision 0 everywhere
        return vec![(0.0, 0.0), (1.0, 0.0)];
    }

    let mut out = Vec::with_capacity(n + 1);
    out.push((0.0, 1.0));

    let mut cum_pos = 0usize;
    let mut cum_total = 0usize;

    let mut i = 0;
    while i < n {
        let score = pairs[i].0;
        let mut group_pos = 0;
        let mut group_total = 0;

        while i < n && score_equal(pairs[i].0, score) {
            if pairs[i].1 == 1 {
                group_pos += 1;
            }
            group_total += 1;
            i += 1;
        }

        cum_pos += group_pos;
        cum_total += group_total;

        let recall = cum_pos as f64 / npos as f64;
        let precision = cum_pos as f64 / cum_total as f64;
        out.push((recall, precision));
    }

    out
}

/// Area under Precision-Recall curve using trapezoid rule on ELKI-style PR curve.
pub fn precision_recall_curve<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let curve = pr_curve(scores, labels);
    let mut area = 0.0;
    for w in curve.windows(2) {
        let (r0, p0) = w[0];
        let (r1, p1) = w[1];
        area += (r1 - r0) * (p0 + p1) * 0.5;
    }
    area
}

pub fn auprc<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    precision_recall_curve(scores, labels)
}

/// Adjusted area under precision-recall curve.
pub fn adjusted_auprc<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let npos = labels.iter().filter(|&&l| l.into() != 0).count() as f64;
    let n = scores.len() as f64;
    super::adjusted_value(auprc(scores, labels), npos / n)
}

#[cfg(test)]
mod tests {
    use super::auprc;

    #[test]
    fn test_auprc_ties() {
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];
        let v = auprc(&scores, &labels);
        assert!((v - 0.7944444444444444).abs() < 1e-12);
    }
}
