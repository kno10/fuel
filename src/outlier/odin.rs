use crate::api::query::IndexQuery;
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

/// Compute ODIN outlier scores based on in-degree of the k‑NN graph.
///
/// For each point the `k` nearest neighbors are found (excluding the point
/// itself), and each neighbor's in-degree is incremented.  Points with a
/// small in-degree are considered outliers (inverted scoring semantics).
///
/// The algorithm returns normalized in-degree: `in_degree / k_effective`.
/// `OutlierMetadata.ascending` is set to `true` to indicate inverted order
/// (low scores are more anomalous).
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn outlier_detection_independence_neighbor<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize,
) -> Result<OutlierResult<F>, String>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    assert!(k > 0, "k must be greater than 0");

    let size = data.len();
    let k_effective = k.min(size.saturating_sub(1));

    if k_effective == 0 {
        return Ok(make_outlier_result(
            vec![F::zero(); size],
            "ODIN",
            true,
            F::zero(),
            F::zero(),
            F::infinity(),
        ));
    }

    let num_threads = {
        #[cfg(feature = "parallel")]
        {
            rayon::current_num_threads().max(1)
        }
        #[cfg(not(feature = "parallel"))]
        {
            1
        }
    };
    let chunk_rows = size.div_ceil(num_threads).max(1);

    let partial_indegrees: Vec<Vec<usize>> = (0..num_threads).par_try_map(|thread_idx| {
        let start = thread_idx * chunk_rows;
        let end = (start + chunk_rows).min(size);
        let mut local = vec![0usize; size];

        for idx in start..end {
            crate::poll_interrupted()?;
            let mut query = data.query();
            query.set_index(idx);
            let neighbors = tree
                .search_knn(&query, k_effective + 1)
                .into_iter()
                .map(|neighbor| neighbor.index)
                .filter(|&neighbor_idx| neighbor_idx != idx);
            for neighbor_idx in neighbors {
                if let Some(d) = local.get_mut(neighbor_idx) {
                    *d += 1;
                }
            }
        }
        Ok(local)
    })?;

    let mut indegree = vec![0usize; size];
    for local in partial_indegrees {
        for (i, count) in local.into_iter().enumerate() {
            indegree[i] += count;
        }
    }

    let inc = 1.0 / (k_effective as f64);
    let scores: Vec<F> =
        indegree.into_iter().map(|d| F::from_f64((d as f64) * inc).unwrap_or(F::zero())).collect();

    Ok(make_outlier_result(scores, "ODIN", true, F::zero(), F::zero(), F::infinity()))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn odin_ranks_remote_point_highest() {
        let points =
            vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.1], vec![0.1, 0.1], vec![6.0, 6.0]];

        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let results = outlier_detection_independence_neighbor(&tree, &data, 2).unwrap();
        assert_eq!(results.scores.len(), points.len());
        assert!(results.scores[4] < results.scores[0]);
    }

    #[test]
    fn odin_single_point_zero() {
        let points = vec![vec![1.0, 2.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(11);
        let tree: VPTree<f64> = VPTree::new(&data, 1, &mut rng);

        let results = outlier_detection_independence_neighbor(&tree, &data, 1).unwrap();
        assert_eq!(results.scores.len(), 1);
        assert_eq!(results.scores[0], 0.0);
        assert_eq!(results.metadata.label, "ODIN");
    }

    #[test]
    fn odin_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = outlier_detection_independence_neighbor(&tree, &data, 10).unwrap();
        let reference = load_reference_scores();
        let expected = reference.get("ODIN-10").expect("No reference for ODIN-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "ODIN-10",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("ODIN-10", &result.scores, expected, 1e-6);
    }
}
