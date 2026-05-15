use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

pub fn k_nearest_neighbors_distance_deviation<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize,
) -> Result<OutlierResult<F>, String>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    let k_effective = k.min(size.saturating_sub(1));

    let neighborhoods: Vec<Vec<(usize, F)>> =
        for_each_knn(tree, data, k_effective, false, |_idx, neighbors| neighbors)?;
    let knn_distances: Vec<F> =
        neighborhoods.iter().map(|n| n.last().map(|(_, d)| *d).unwrap_or(F::zero())).collect();

    let scores: Vec<F> = (0..size).par_map(|i| {
        let d = knn_distances[i].to_f64().unwrap_or(0.0);
        let neighbor_idx = neighborhoods[i].last().map(|(idx, _)| *idx);
        let nd = neighbor_idx.map(|j| knn_distances[j].to_f64().unwrap_or(0.0)).unwrap_or(0.0);

        let sc = if nd > 0.0 {
            d / nd
        } else if d > 0.0 {
            f64::INFINITY
        } else {
            1.0
        };

        F::from_f64(sc).unwrap_or(F::zero())
    });

    Ok(make_outlier_result(scores, "kNNDensityDiff", false, F::zero(), F::zero(), F::infinity()))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;

    #[test]
    fn knndd_test() {
        let points = vec![vec![0.0], vec![0.1], vec![1.0], vec![100.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = k_nearest_neighbors_distance_deviation(&tree, &data, 1).unwrap();
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }

    #[test]
    fn knndd_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = k_nearest_neighbors_distance_deviation(&tree, &data, 10).unwrap();
        let reference = load_reference_scores();
        let expected = reference.get("KNNDD-10").expect("No reference for KNNDD-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "KNNDD-10",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("KNNDD-10", &result.scores, expected, 1e-6);
    }
}
