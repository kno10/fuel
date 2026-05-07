use crate::evaluation::outlier::{score_equal, sort_score_label};

fn discount(rank: usize) -> f64 { 1.0 / ((rank as f64 + 1.0).log2()) }

fn ideal_dcg(npos: usize) -> f64 { (1..=npos).map(discount).sum() }

/// DCG in presence of ties: expected position within tied group.
pub fn discounted_cumulative_gain<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let pairs = sort_score_label(scores, labels);
    let n = pairs.len();
    if n == 0 {
        return 0.0;
    }

    let mut result = 0.0;
    let mut idx = 0usize;
    let mut rank_offset = 0usize;

    while idx < n {
        let score = pairs[idx].0;
        let mut group_pos = 0usize;
        let mut group_size = 0usize;

        while idx < n {
            if score_equal(pairs[idx].0, score) {
                if pairs[idx].1 == 1 {
                    group_pos += 1;
                }
                group_size += 1;
                idx += 1;
            } else {
                break;
            }
        }

        if group_pos > 0 {
            let discount_sum: f64 =
                ((rank_offset + 1)..=(rank_offset + group_size)).map(discount).sum();
            let per_positive = discount_sum / (group_size as f64);
            result += per_positive * (group_pos as f64);
        }

        rank_offset += group_size;
    }

    result
}

pub fn dcg<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    discounted_cumulative_gain(scores, labels)
}

/// Normalized DCG computed as DCG / ideal DCG.
pub fn normalized_discounted_cumulative_gain<
    F: Copy + Into<f64> + PartialOrd,
    L: Copy + Into<u8>,
>(
    scores: &[F], labels: &[L],
) -> f64 {
    let npos = labels.iter().filter(|&&l| if l.into() != 0 { 1 } else { 0 } == 1).count();
    if npos == 0 {
        return 0.0;
    }
    let ideal = ideal_dcg(npos);
    if ideal <= 0.0 {
        return 0.0;
    }
    (dcg(scores, labels) / ideal).clamp(0.0, 1.0)
}

/// Adjusted DCG, computed from normalized DCG with expected random baseline.
fn expected_ndcg(npos: usize, n: usize) -> f64 {
    if n == 0 || npos == 0 {
        return 0.0;
    }
    let ideal = (1..=npos).map(|rank| 1.0 / ((rank as f64 + 1.0).log2())).sum::<f64>();
    if ideal <= 0.0 {
        return 0.0;
    }
    let expected_dcg = (npos as f64 / n as f64)
        * (1..=n).map(|rank| 1.0 / ((rank as f64 + 1.0).log2())).sum::<f64>();
    expected_dcg / ideal
}

pub fn adjusted_dcg<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let n = scores.len();
    let ndcg = normalized_discounted_cumulative_gain(scores, labels);
    let expected = expected_ndcg(labels.iter().filter(|&&l| l.into() != 0).count(), n);
    if expected >= 1.0 {
        0.0
    } else {
        ((ndcg - expected) / (1.0 - expected)).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{adjusted_dcg, dcg, normalized_discounted_cumulative_gain};

    #[test]
    fn test_dcg_ndcg_ties() {
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];
        // ideal gains for 3 positives: 1/log2(2)+1/log2(3)+1/log2(4)
        let ideal = 1.0 / 2f64.log2() + 1.0 / 3f64.log2() + 1.0 / 4f64.log2();
        let v = normalized_discounted_cumulative_gain(&scores, &labels);
        // with ties the top score group (0.6) includes 1 positive among 2 items.
        assert!(v > 0.0 && v <= 1.0);
        assert!((dcg(&scores, &labels) / ideal - v).abs() < 1e-12);
    }

    #[test]
    fn test_ndcg_clamps_to_one() {
        let scores = [0.9, 0.8, 0.7, 0.6];
        let labels = [1u8, 1, 1, 1];
        let v = normalized_discounted_cumulative_gain(&scores, &labels);
        assert_eq!(v, 1.0);
    }

    #[test]
    fn test_adjusted_dcg_clamps_to_one() {
        let scores = [0.9, 0.8, 0.7, 0.6, 0.5];
        let labels = [1u8, 1, 1, 0, 0];
        let v = adjusted_dcg(&scores, &labels);
        assert_eq!(v, 1.0);
    }

    #[test]
    fn test_dcg_nan_score_does_not_loop_forever() {
        let scores = [0.3, f64::NAN, 0.6];
        let labels = [1u8, 0, 1];
        let _ = dcg(&scores, &labels);
        let _ = normalized_discounted_cumulative_gain(&scores, &labels);
    }
}
