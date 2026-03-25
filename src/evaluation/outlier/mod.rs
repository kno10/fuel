pub mod average_precision;
pub mod discounted_cumulative_gain;
pub mod maximum_f1;
pub mod precision_at_k;
pub mod precision_recall_curve;
pub mod precision_recall_gain;
pub mod receiver_operating_curve;

use std::cmp::Ordering;

/// Shared sorter for (`score`, `label`) pairs in descending score order.
/// Labels are mapped to binary 0/1 (nonzero -> 1).
pub fn sort_score_label<F: Copy + Into<f64> + PartialOrd, L: Copy + Into<u8>>(
    scores: &[F], labels: &[L],
) -> Vec<(f64, u8)> {
    assert_eq!(scores.len(), labels.len(), "scores and labels mismatch");

    let mut pairs: Vec<(f64, u8)> = scores
        .iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s.into(), if l.into() != 0 { 1 } else { 0 }))
        .collect();

    pairs.sort_by(|a, b| match a.0.partial_cmp(&b.0) {
        Some(Ordering::Less) => Ordering::Greater,
        Some(Ordering::Greater) => Ordering::Less,
        _ => Ordering::Equal,
    });

    pairs
}
