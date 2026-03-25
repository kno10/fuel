use crate::evaluation::outlier::sort_score_label;

/// Area under ROC with tie handling consistent with ELKI: pairs in tied score groups get 0.5.
pub fn receiver_operating_curve<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let pairs = sort_score_label(scores, labels);

    let n = pairs.len();
    if n == 0 {
        return 0.0;
    }

    let npos = pairs.iter().filter(|(_, l)| *l == 1).count();
    let nneg = n - npos;
    if npos == 0 || nneg == 0 {
        return 0.5;
    }

    let mut tp = 0usize;
    let mut auctotal = 0.0;

    let mut i = 0;
    while i < n {
        let score = pairs[i].0;
        let mut group_pos = 0usize;
        let mut group_neg = 0usize;

        while i < n && pairs[i].0 == score {
            if pairs[i].1 == 1 {
                group_pos += 1;
            } else {
                group_neg += 1;
            }
            i += 1;
        }

        auctotal += tp as f64 * group_neg as f64 + (group_pos as f64 * group_neg as f64) * 0.5;

        tp += group_pos;
    }

    auctotal / (npos as f64 * nneg as f64)
}

pub fn auc<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    receiver_operating_curve(scores, labels)
}

#[cfg(test)]
mod tests {
    use super::auc;

    #[test]
    fn test_auc_ties() {
        // Score order: 0.9(P), 0.6(P), 0.6(N), 0.3(N), 0.1(P)
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];
        let v = auc(&scores, &labels);
        assert!((v - 0.5833333333333333).abs() < 1e-12);
    }
}
