use crate::evaluation::outlier::sort_score_label;

/// Expected average precision in presence of ties, as in ELKI.
#[allow(clippy::float_cmp, clippy::cast_precision_loss)]
// exact equality is intentional for tie grouping on scores and using f64 to aggregate ratios
pub fn average_precision<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
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

    let mut cum_pos = 0usize;
    let mut cum_total = 0usize;
    let mut ap = 0.0;

    let mut i = 0;
    while i < n {
        let score = pairs[i].0;
        let mut gp = 0usize;
        let mut gs = 0usize;

        while i < n && pairs[i].0 == score {
            if pairs[i].1 == 1 {
                gp += 1;
            }
            gs += 1;
            i += 1;
        }

        if gp > 0 {
            for group_pos_idx in 1..=gs {
                let p_pos = gp as f64 / gs as f64;
                let expected_tp = if gs == 1 {
                    cum_pos as f64 + 1.0
                } else {
                    cum_pos as f64
                        + 1.0
                        + (group_pos_idx as f64 - 1.0) * (gp as f64 - 1.0) / (gs as f64 - 1.0)
                };
                let rank = cum_total as f64 + group_pos_idx as f64;
                let expected_precision_given_pos = expected_tp / rank;
                ap += (1.0 / npos as f64) * p_pos * expected_precision_given_pos;
            }
        }

        cum_pos += gp;
        cum_total += gs;
    }

    ap
}

#[cfg(test)]
mod tests {
    use super::average_precision;

    #[test]
    fn test_average_precision_ties() {
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];
        let v = average_precision(&scores, &labels);
        assert!((v - 0.81111111).abs() < 1e-7);
    }
}
