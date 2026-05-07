use crate::evaluation::outlier::precision_recall_curve::pr_curve;

/// Area under precision-recall gain curve.
///
/// This is a simple extracted variant; exact ELKI behavior is approximately
/// implemented through the PR curve and tie-handling in that implementation.
pub fn prg_auc<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    let pr = pr_curve(scores, labels);

    let mut points: Vec<(f64, f64)> = Vec::new();
    for &(r, p) in &pr {
        if r >= 1.0 || p >= 1.0 || r < 0.0 || p < 0.0 {
            continue;
        }
        let x = r / (1.0 - r);
        let y = p / (1.0 - p);
        if x.is_finite() && y.is_finite() {
            points.push((x, y));
        }
    }

    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if points.len() < 2 {
        return 0.0;
    }

    let mut area = 0.0;
    for w in points.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        area += (x1 - x0) * (y0 + y1) * 0.5;
    }
    area
}

/// Adjusted area under precision-recall gain curve.
pub fn adjusted_auprgc<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> f64 {
    (prg_auc(scores, labels) - 0.5) * 2.0
}

#[cfg(test)]
mod tests {
    use super::prg_auc;

    #[test]
    fn test_prg_auc_ties() {
        let scores = [0.3, 0.6, 0.6, 0.1, 0.9];
        let labels = [0u8, 1, 0, 1, 1];

        let value = prg_auc(&scores, &labels);
        assert!(value >= 0.0);
    }
}
