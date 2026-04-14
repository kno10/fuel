use crate::Float;
use crate::cluster::hierarchical::MergeHistory;
use crate::cluster::hierarchical::common::UnionFind;
use crate::cluster::hierarchical::extraction::common::compress_labels;

/// Extract flat cluster labels by cutting a merge history to at least
/// `min_clusters` clusters.
///
/// This mirrors ELKI's tie handling: if the distance at the cut boundary is
/// tied with previous merge distances, all tied merges are excluded so the
/// resulting cluster count can be larger than `min_clusters`.
///
/// Labels are contiguous `0..k-1` in first-occurrence order.
#[must_use]
pub fn cut_dendrogram_by_number_of_clusters<F: Float>(
    history: &MergeHistory<F>, min_clusters: usize,
) -> Vec<usize> {
    assert!(min_clusters > 0, "min_clusters must be positive");

    let n = history.len() + 1;
    if min_clusters >= n {
        return (0..n).collect();
    }

    let split = find_split(history, n, min_clusters);
    cut_by_split(history, n, split)
}

fn find_split<F: Float>(history: &MergeHistory<F>, n: usize, min_clusters: usize) -> usize {
    if n <= min_clusters {
        return 0;
    }

    let mut split = n - min_clusters;
    if split >= history.len() {
        return history.len();
    }
    let stop_dist = history.get(split).unwrap().distance();

    while split > 0 && history.get(split - 1).unwrap().distance() == stop_dist {
        split -= 1;
    }

    split
}

pub(crate) fn cut_by_split<F: Float>(
    history: &MergeHistory<F>, n: usize, split: usize,
) -> Vec<usize> {
    assert_eq!(history.len() + 1, n, "history length does not match n");
    assert!(split <= history.len(), "split out of range");

    let mut uf = UnionFind::new(n);
    let mut cluster_rep = Vec::with_capacity(n + split);
    cluster_rep.extend(0..n);

    for (step, merge) in history.iter().take(split).enumerate() {
        let rep1 = uf.find(cluster_rep[merge.idx1()]);
        let rep2 = uf.find(cluster_rep[merge.idx2()]);
        let (_, cid) = uf.union(rep1, rep2);
        cluster_rep.push(cid);

        // Keep consistency with SciPy/AGNES cluster-id numbering.
        debug_assert_eq!(n + step, cluster_rep.len() - 1);
    }

    let mut roots = Vec::with_capacity(n);
    for point in 0..n {
        roots.push(uf.find(point));
    }
    compress_labels(&roots)
}

#[cfg(test)]
mod tests {
    use super::cut_dendrogram_by_number_of_clusters;
    use crate::cluster::hierarchical::{Merge, MergeHistory, SingleLinkage, agnes};

    #[test]
    fn cut_by_cluster_count_produces_expected_groups() {
        let d = vec![1.0, 2.0, 3.0, 1.5, 2.5, 1.0];
        let cm = crate::CondensedDistanceMatrix::new_from_condensed(d, 4, false);
        let history = agnes(&cm, SingleLinkage);

        let labels_one = cut_dendrogram_by_number_of_clusters(&history, 1);
        assert_eq!(labels_one, vec![0, 0, 0, 0]);

        let labels_two = cut_dendrogram_by_number_of_clusters(&history, 2);
        assert_eq!(labels_two, vec![0, 0, 1, 1]);

        let labels_three = cut_dendrogram_by_number_of_clusters(&history, 3);
        assert_eq!(labels_three, vec![0, 1, 2, 3]);

        let labels_four = cut_dendrogram_by_number_of_clusters(&history, 4);
        assert_eq!(labels_four, vec![0, 1, 2, 3]);
    }

    #[test]
    fn tie_handling_may_return_more_clusters() {
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 5, distance: 2.0, size: 4, prototype: usize::MAX },
        ]
        .into();

        let labels = cut_dendrogram_by_number_of_clusters(&history, 3);
        // Both first merges are tied at the split distance and therefore
        // removed from the lower part: all points remain singleton clusters.
        assert_eq!(labels, vec![0, 1, 2, 3]);
    }
}
