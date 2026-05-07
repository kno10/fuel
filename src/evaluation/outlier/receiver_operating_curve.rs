use crate::evaluation::outlier::{score_equal, sort_score_label};

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

        while i < n && score_equal(pairs[i].0, score) {
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

pub fn auroc<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    receiver_operating_curve(scores, labels)
}

/// Adjusted area under ROC curve, with random baseline mapped to 0.
pub fn adjusted_auroc<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    2.0 * auroc(scores, labels) - 1.0
}

#[cfg(test)]
mod tests {
    use super::auroc;

    #[test]
    fn test_auroc_ties() {
        // Score order: 0.9(P), 0.6(P), 0.6(N), 0.3(N), 0.1(P)
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];
        let v = auroc(&scores, &labels);
        assert!((v - 0.5833333333333333).abs() < 1e-12);
    }
}
