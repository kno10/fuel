use crate::Float;
use crate::cluster::hierarchical::MergeHistory;
use crate::cluster::hierarchical::extraction::by_number_of_clusters::cut_by_split;

/// Extract flat cluster labels by cutting the dendrogram at `threshold`.
///
/// Merges with distance strictly less than `threshold` are retained.
/// Labels are contiguous `0..k-1` in first-occurrence order.
#[must_use]
pub fn cut_dendrogram_by_height<F: Float>(history: &MergeHistory<F>, threshold: F) -> Vec<usize> {
    let n = history.len() + 1;
    let split = find_split(history, threshold);
    cut_by_split(history, n, split)
}

fn find_split<F: Float>(history: &MergeHistory<F>, threshold: F) -> usize {
    let mut split = history.len();
    while split > 1 && threshold <= history.get(split - 1).unwrap().distance {
        split -= 1;
    }
    split
}

#[cfg(test)]
mod tests {
    use super::cut_dendrogram_by_height;
    use crate::cluster::hierarchical::{SingleLinkage, agnes};

    #[test]
    fn cut_by_height_matches_expected_partition() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let cm = crate::CondensedDistanceMatrix::new_from_condensed(d, 4, false);
        let history = agnes(&cm, SingleLinkage);

        let labels = cut_dendrogram_by_height(&history, 1.1);
        assert_eq!(labels, vec![0, 0, 1, 1]);

        let labels_high = cut_dendrogram_by_height(&history, 2.0);
        assert_eq!(labels_high, vec![0, 0, 0, 0]);
    }
}
