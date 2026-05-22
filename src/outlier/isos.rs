use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::outlier::sos;
use crate::{DistanceData, Float, KnnSearch};

const DEFAULT_PHI: f64 = 0.01;

fn transform_scores(scores: &mut [f64], log_perp: f64, phi: f64) -> (f64, f64) {
    let adj = (1.0 - phi) / phi;
    let mut min_score = f64::INFINITY;
    let mut max_score = f64::NEG_INFINITY;
    for score in scores.iter_mut() {
        let or = (-*score * log_perp).exp() * adj;
        let s = 1.0 / (1.0 + or);
        *score = s;
        if s < min_score {
            min_score = s;
        }
        if s > max_score {
            max_score = s;
        }
    }
    if min_score.is_infinite() && max_score.is_infinite() {
        min_score = 0.0;
        max_score = 0.0;
    }
    (min_score, max_score)
}

fn adjust_distances<F>(neighbors: &[(usize, F)], max_distance: f64, id: f64) -> Vec<f64>
where
    F: Float,
{
    let scalelin = 1.0 / max_distance;
    let scaleexp = id * 0.5;
    neighbors
        .iter()
        .map(|&(_, dist)| {
            let d = dist.to_f64().unwrap_or(f64::INFINITY);
            (d * scalelin).powf(scaleexp)
        })
        .collect()
}

/// Intrinsic Stochastic Outlier Selection (ISOS).
///
/// The `E` estimator is used for local intrinsic dimensionality.
pub fn intrinsic_stochastic_outlier_selection<'a, S, D, F, E>(
    tree: &S, data: &'a D, k: usize,
) -> Result<OutlierResult<F>, String>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    E: crate::intrinsicdimensionality::KNNIDEstimator,
{
    let size = data.len();
    if size == 0 {
        return Ok(make_outlier_result(
            Vec::new(),
            "ISOS",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        ));
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return Ok(make_outlier_result(
            vec![F::zero(); size],
            "ISOS",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        ));
    }

    let perplexity = (k as f64) / 3.0;
    let log_perp = if perplexity > 1.0 { perplexity.ln() } else { 0.1 };

    let mut scores = vec![1.0f64; size];

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh)?;
    for (idx, neighbors) in neighborhoods.iter().enumerate().take(size) {
        if neighbors.is_empty() {
            continue;
        }

        let neighbors_no_self: Vec<(usize, F)> =
            neighbors.iter().cloned().filter(|(neighbor_id, _)| *neighbor_id != idx).collect();

        let all_distances: Vec<f64> =
            neighbors.iter().map(|&(_, d)| d.to_f64().unwrap_or(f64::INFINITY)).collect();

        let id = E::estimate_from_knn(tree, data, idx, k_effective + 1);

        let probs = if id.is_finite() {
            let max_distance = neighbors_no_self
                .iter()
                .map(|&(_, d)| d.to_f64().unwrap_or(f64::INFINITY))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let adjusted = adjust_distances(&neighbors_no_self, max_distance, id);
            sos::compute_pi(&adjusted, perplexity)
        } else {
            sos::compute_pi(&all_distances, perplexity)
        };

        let sum_p: f64 = probs.iter().sum();
        if sum_p > 0.0 {
            for (i, &(neighbor_id, _)) in neighbors_no_self.iter().enumerate() {
                let v = probs[i] / sum_p;
                if v.is_nan() || v <= 0.0 {
                    break;
                }
                scores[neighbor_id] += (-v).ln_1p();
            }
        }
    }

    let (_min_score, _max_score) = transform_scores(&mut scores, log_perp, DEFAULT_PHI);

    let final_scores: Vec<F> =
        scores.into_iter().map(|v| F::from_f64(v).unwrap_or(F::zero())).collect();

    Ok(make_outlier_result(final_scores, "ISOS", false, F::zero(), F::zero(), F::infinity()))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::intrinsicdimensionality::HillID;
    use crate::outlier::common::*;

    #[test]
    fn isos_20_hill_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result =
            intrinsic_stochastic_outlier_selection::<_, _, _, HillID>(&tree, &data, 20).unwrap();
        let reference = load_reference_scores();
        let expected = reference.get("ISOS-20-Hill").expect("No reference for ISOS-20-Hill");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "ISOS-20-Hill",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("ISOS-20-Hill", &result.scores, expected, 1e-6);
    }

    #[test]
    fn isos_10_aggregated_hill_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = intrinsic_stochastic_outlier_selection::<
            _,
            _,
            _,
            crate::intrinsicdimensionality::AggregatedHillID,
        >(&tree, &data, 10)
        .unwrap();
        let reference = load_reference_scores();
        let expected = reference.get("ISOS-10").expect("No reference for ISOS-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "ISOS-10",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("ISOS-10", &result.scores, expected, 1e-6);
    }
}
