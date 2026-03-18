use num_traits::{Float, FromPrimitive};

use super::common::{OutlierScoreEntry, sort_outlier_scores_ascending};
#[cfg(test)]
use crate::distance::EuclideanDistance;
use crate::{DistanceData, KnnSearch};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdinOutlierScore<F: Float> {
    pub index: usize,
    pub score: F,
}

impl<F: Float> OutlierScoreEntry<F> for OdinOutlierScore<F> {
    fn index(&self) -> usize {
        self.index
    }

    fn score(&self) -> F {
        self.score
    }
}

/// Compute ODIN outlier scores based on in-degree of the k‑NN graph.
///
/// For each point the `k` nearest neighbors are found (excluding the point
/// itself).  We increment the in-degree counter of each neighbor.  Points
/// with a small in-degree are considered outliers.  The returned score is
/// simply `1/(in_degree+1)` so that smaller degrees map to larger scores; the
/// vector is sorted with highest scores first.
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn odin_outlier_scores<S, D, F>(
    tree: &S,
    data: D,
    k: usize,
) -> Vec<OdinOutlierScore<F>>
where
    F: Float + FromPrimitive,
    D: DistanceData<F>,
    S: KnnSearch<F, D>,
{
    assert!(k > 0, "k must be greater than 0");

    let size = data.size();
    let k_effective = k.min(size.saturating_sub(1));

    // if there are no neighbors we just return equal scores
    if k_effective == 0 {
        return (0..size)
            .map(|idx| OdinOutlierScore {
                index: idx,
                score: F::one(),
            })
            .collect();
    }

    let mut indegree = vec![0u32; size];

    for idx in data.iter() {
        let neighbors = tree
            .search_knn_by_index(&data, idx, (k_effective + 1).min(size))
            .into_iter()
            .filter(|neighbor| neighbor.index != idx)
            .take(k_effective);

        for neigh in neighbors {
            indegree[neigh.index] += 1;
        }
    }

    let mut scores: Vec<OdinOutlierScore<F>> = Vec::with_capacity(size);

    for (idx, &deg_i) in indegree.iter().enumerate().take(size) {
        let score = F::from_f64(1.0 / (deg_i as f64 + 1.0)).unwrap_or(F::zero());
        scores.push(OdinOutlierScore { index: idx, score });
    }

    sort_outlier_scores_ascending(&mut scores);

    scores
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::TableWithDistance;
    use crate::vptree::VPTree;

    use super::*;

    #[test]
    fn odin_ranks_remote_point_highest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![6.0, 6.0],
        ];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let scores = odin_outlier_scores(&tree, &data, 2);
        assert_eq!(scores.len(), points.len());
        let last_index = scores.len() - 1;
        assert_eq!(scores[last_index].index, 4);
        assert!(scores[last_index].score > scores[last_index - 1].score);
    }

    #[test]
    fn odin_single_point_zero() {
        let points = vec![vec![1.0, 2.0]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(11);
        let tree: VPTree<f64> = VPTree::new(&data, 1, &mut rng);

        let scores = odin_outlier_scores(&tree, &data, 1);
        assert_eq!(
            scores,
            vec![OdinOutlierScore {
                index: 0,
                score: 1.0
            }]
        );
    }
}
