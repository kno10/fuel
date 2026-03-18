use std::cmp::Ordering;

use num_traits::Float;

/// A minimal trait used by helper utilities that operate on outlier scores.
pub trait OutlierScoreEntry<F: Float> {
    /// The point index the score corresponds to.
    fn index(&self) -> usize;

    /// The numeric score used for ranking.
    fn score(&self) -> F;
}

/// Sort a slice of outlier scores so that higher scores appear first.
pub fn sort_outlier_scores<T: OutlierScoreEntry<F>, F: Float>(scores: &mut [T]) {
    scores.sort_by(|a, b| {
        b.score()
            .partial_cmp(&a.score())
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.index().cmp(&b.index()))
    });
}

/// Sort scores so the smallest values come first while still breaking ties on index.
pub fn sort_outlier_scores_ascending<T: OutlierScoreEntry<F>, F: Float>(scores: &mut [T]) {
    scores.sort_by(|a, b| {
        a.score()
            .partial_cmp(&b.score())
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.index().cmp(&b.index()))
    });
}
