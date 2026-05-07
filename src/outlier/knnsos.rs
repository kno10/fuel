use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::outlier::sos;
use crate::{DistanceData, Float, KnnSearch};

pub fn k_nearest_neighbors_sos<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "kNNSOS",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }
    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::zero(); size],
            "kNNSOS",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let perplexity = (k as f64) / 3.0;
    let log_perp = if perplexity > 1.0 { perplexity.ln() } else { 0.1 };

    let mut scores = vec![1.0f64; size];

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);
    for (_idx, neighbors) in neighborhoods.iter().enumerate().take(size) {
        if neighbors.is_empty() {
            continue;
        }

        let distances: Vec<f64> =
            neighbors.iter().map(|&(_, d)| d.to_f64().unwrap_or(f64::INFINITY)).collect();

        let probs = sos::compute_pi(&distances, perplexity);
        let sum_p: f64 = probs.iter().filter(|&&p| p > 0.0).sum();
        if sum_p > 0.0 {
            for (i, &(neighbor_id, _)) in neighbors.iter().enumerate() {
                let v = probs[i] / sum_p;
                if v.is_nan() || v <= 0.0 {
                    break;
                }
                scores[neighbor_id] += (-v).ln_1p();
            }
        }
    }

    let mut scores_after = scores.clone();
    let adj = (1.0 - 0.01) / 0.01;
    for score in &mut scores_after {
        let or = (-*score * log_perp).exp() * adj;
        *score = 1.0 / (1.0 + or);
    }

    let final_scores: Vec<F> =
        scores_after.into_iter().map(|v| F::from_f64(v).unwrap_or(F::zero())).collect();

    make_outlier_result(final_scores, "kNNSOS", false, F::zero(), F::zero(), F::infinity())
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
    fn knnsos_test() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = k_nearest_neighbors_sos(&tree, &data, 2);
        assert!(results.scores.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));

        let min_score = results.scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = results.scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_score > min_score);
    }

    #[test]
    fn knnsos_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = k_nearest_neighbors_sos(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("KNNSOS-10").expect("No reference for KNNSOS-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "KNNSOS-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("KNNSOS-10", &result.scores, expected, 1e-6);
    }
}
