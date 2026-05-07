use crate::evaluation::outlier::sort_score_label;

/// Maximum F1 score across thresholds with ties handled by group-level values.
pub fn maximum_f1<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let pairs = sort_score_label(scores, labels);

    let n = pairs.len();
    if n == 0 {
        return 0.0;
    }
    let npos = pairs.iter().filter(|(_, l)| *l == 1).count();
    if npos == 0 {
        return 0.0;
    }

    let mut best_f1 = 0.0;
    let mut cum_pos = 0usize;
    let mut cum_total = 0usize;

    let mut i = 0;
    while i < n {
        let score = pairs[i].0;
        let mut group_pos = 0usize;
        let mut group_total = 0usize;

        while i < n && pairs[i].0 == score {
            if pairs[i].1 == 1 {
                group_pos += 1;
            }
            group_total += 1;
            i += 1;
        }

        cum_pos += group_pos;
        cum_total += group_total;

        let precision = (cum_pos as f64) / (cum_total as f64);
        let recall = (cum_pos as f64) / (npos as f64);
        if precision > 0.0 && recall > 0.0 {
            let f1 = 2.0 * precision * recall / (precision + recall);
            if f1 > best_f1 {
                best_f1 = f1;
            }
        }
    }

    best_f1
}

/// Adjusted maximum F1.
pub fn adjusted_maximum_f1<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let npos = labels.iter().filter(|&&l| l.into() != 0).count() as f64;
    let n = scores.len() as f64;
    super::adjusted_value(maximum_f1(scores, labels), npos / n)
}

#[cfg(test)]
mod tests {
    use super::maximum_f1;

    #[test]
    fn test_maximum_f1_ties() {
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];
        let v = maximum_f1(&scores, &labels);
        assert!((v - 0.75).abs() < 1e-12);
    }
}
