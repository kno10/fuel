use crate::Float;
use crate::cluster::hierarchical::MergeHistory;

/// Compute pairwise cophenetic distances from a dendrogram.
///
/// For each pair of original points `(i, j)` with `i > j`, the cophenetic
/// distance is the merge distance at which `i` and `j` first join the same
/// cluster.  Results are stored in a condensed lower-triangular distance
/// matrix of length `n * (n - 1) / 2`, indexed by `i * (i - 1) / 2 + j`.
///
/// The merge history must follow the SciPy linkage convention: original
/// points are `0..n`, merged clusters are numbered from `n` onwards.
pub fn cophenetic_distances<F: Float>(history: &MergeHistory<F>, n: usize) -> Vec<F> {
    if n <= 1 {
        return Vec::new();
    }
    let num_pairs = n * (n - 1) / 2;
    let mut coph = vec![F::zero(); num_pairs];

    // Linked-list member chains for each cluster.
    let num_clusters = n + history.len();
    let mut head = vec![usize::MAX; num_clusters];
    let mut tail = vec![usize::MAX; num_clusters];
    let mut next = vec![usize::MAX; n];
    for i in 0..n {
        head[i] = i;
        tail[i] = i;
    }

    for (k, merge) in history.iter().enumerate() {
        let new_id = n + k;
        let d = merge.distance;
        // Set cophenetic distance for every cross-pair.
        let mut a = head[merge.idx1];
        while a != usize::MAX {
            let mut b = head[merge.idx2];
            while b != usize::MAX {
                let (big, small) = if a > b { (a, b) } else { (b, a) };
                coph[big * (big - 1) / 2 + small] = d;
                b = next[b];
            }
            a = next[a];
        }
        // Concatenate the two member chains into the new cluster.
        if head[merge.idx1] != usize::MAX {
            if head[merge.idx2] != usize::MAX {
                next[tail[merge.idx1]] = head[merge.idx2];
                tail[merge.idx1] = tail[merge.idx2];
            }
            head[new_id] = head[merge.idx1];
            tail[new_id] = tail[merge.idx1];
        } else {
            head[new_id] = head[merge.idx2];
            tail[new_id] = tail[merge.idx2];
        }
    }

    coph
}

/// Pearson correlation between two dendrograms' cophenetic distances.
///
/// Builds the full cophenetic distance vectors for both merge histories
/// and returns their Pearson correlation coefficient.  A value of 1.0
/// means the dendrograms induce identical cophenetic distances.
pub fn cophenetic_correlation(
    base: &MergeHistory<f64>, other: &MergeHistory<f64>, n: usize,
) -> f64 {
    let x = cophenetic_distances(base, n);
    let y = cophenetic_distances(other, n);
    pearson_correlation(&x, &y)
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let len = x.len();
    if len == 0 {
        return 1.0;
    }
    let nf = len as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|&a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|&b| b * b).sum();
    let numerator = nf * sum_xy - sum_x * sum_y;
    let denom_sq = (nf * sum_x2 - sum_x * sum_x) * (nf * sum_y2 - sum_y * sum_y);
    if denom_sq <= 0.0 {
        return 0.0;
    }
    numerator / denom_sq.sqrt()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::cluster::hierarchical::Merge;

    /// Build a simple 3-point dendrogram:
    ///   merge 0: {0} + {1} at distance 1.0 -> cluster 3
    ///   merge 1: {3} + {2} at distance 3.0 -> cluster 4
    ///
    /// Expected cophenetic distances:
    ///   coph(1, 0) = 1.0
    ///   coph(2, 0) = 3.0
    ///   coph(2, 1) = 3.0
    fn three_point_history() -> MergeHistory<f64> {
        vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 3, idx2: 2, distance: 3.0, size: 3, prototype: usize::MAX },
        ]
        .into()
    }

    #[test]
    fn cophenetic_distances_three_points() {
        let history = three_point_history();
        let coph = cophenetic_distances(&history, 3);
        assert_eq!(coph.len(), 3);
        // index: i*(i-1)/2 + j
        assert_eq!(coph[0], 1.0); // (1,0)
        assert_eq!(coph[1], 3.0); // (2,0)
        assert_eq!(coph[2], 3.0); // (2,1)
    }

    #[test]
    fn cophenetic_distances_four_points_chain() {
        // Chain dendrogram: 0-1 at 1.0, then +2 at 2.0, then +3 at 5.0
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 2, distance: 2.0, size: 3, prototype: usize::MAX },
            Merge { idx1: 5, idx2: 3, distance: 5.0, size: 4, prototype: usize::MAX },
        ]
        .into();
        let coph = cophenetic_distances(&history, 4);
        assert_eq!(coph.len(), 6);
        // (1,0)=1.0, (2,0)=2.0, (2,1)=2.0, (3,0)=5.0, (3,1)=5.0, (3,2)=5.0
        assert_eq!(coph[0], 1.0);
        assert_eq!(coph[1], 2.0);
        assert_eq!(coph[2], 2.0);
        assert_eq!(coph[3], 5.0);
        assert_eq!(coph[4], 5.0);
        assert_eq!(coph[5], 5.0);
    }

    #[test]
    fn cophenetic_distances_balanced_tree() {
        // Balanced: {0,1} at 1.0, {2,3} at 1.5, then merge at 4.0
        let history: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 2, idx2: 3, distance: 1.5, size: 2, prototype: usize::MAX },
            Merge { idx1: 4, idx2: 5, distance: 4.0, size: 4, prototype: usize::MAX },
        ]
        .into();
        let coph = cophenetic_distances(&history, 4);
        // (1,0)=1.0, (2,0)=4.0, (2,1)=4.0, (3,0)=4.0, (3,1)=4.0, (3,2)=1.5
        assert_eq!(coph[0], 1.0);
        assert_eq!(coph[1], 4.0);
        assert_eq!(coph[2], 4.0);
        assert_eq!(coph[3], 4.0);
        assert_eq!(coph[4], 4.0);
        assert_eq!(coph[5], 1.5);
    }

    #[test]
    fn cophenetic_correlation_identical_dendrograms() {
        let h = three_point_history();
        let r = cophenetic_correlation(&h, &h, 3);
        assert!((r - 1.0).abs() < 1e-12, "identical dendrograms should give r=1.0, got {r}");
    }

    #[test]
    fn cophenetic_correlation_different_dendrograms() {
        let h1: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 1, distance: 1.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 3, idx2: 2, distance: 3.0, size: 3, prototype: usize::MAX },
        ]
        .into();
        // Different merge order: {0,2} first, then +1
        let h2: MergeHistory<f64> = vec![
            Merge { idx1: 0, idx2: 2, distance: 2.0, size: 2, prototype: usize::MAX },
            Merge { idx1: 3, idx2: 1, distance: 4.0, size: 3, prototype: usize::MAX },
        ]
        .into();
        let r = cophenetic_correlation(&h1, &h2, 3);
        // h1: [1, 3, 3], h2: [4, 2, 4] -> r should be negative
        assert!(r < 0.0, "expected negative correlation, got {r}");
    }
}
