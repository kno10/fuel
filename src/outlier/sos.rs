use crate::api::IndexQuery;
use crate::outlier::common::{OutlierResult, make_outlier_result, perplexity_to_entropy};
use crate::{DistanceData, Float, KnnSearch};

const PERPLEXITY_ERROR: f64 = 1e-5;
const PERPLEXITY_MAXITER: usize = 50;

fn estimate_initial_beta(distances: &[f64], perplexity: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0;
    for &d in distances.iter() {
        if d.is_finite() && d > 0.0 {
            sum += d;
            count += 1;
        }
    }
    if sum > 0.0 && sum < f64::INFINITY && count > 1 {
        0.5 / sum * perplexity * ((count - 1) as f64)
    } else {
        1.0
    }
}

fn compute_h(distances: &[f64], beta: f64) -> (f64, Vec<f64>) {
    let mut p = vec![0.0; distances.len()];
    let mut sum_p = 0.0;
    for (i, &d) in distances.iter().enumerate() {
        let val = (-d * beta).exp();
        p[i] = val;
        sum_p += val;
    }
    if sum_p <= 0.0 {
        return (f64::NEG_INFINITY, p);
    }

    let inv_sum_p = 1.0 / sum_p;
    let mut weighted_sum = 0.0;
    for (i, &d) in distances.iter().enumerate() {
        p[i] *= inv_sum_p;
        weighted_sum += d * p[i];
    }

    let h = sum_p.ln() + beta * weighted_sum;
    (h, p)
}

pub(crate) fn compute_pi(distances: &[f64], perplexity: f64) -> Vec<f64> {
    let log_perp = perplexity_to_entropy(perplexity);
    let mut beta = estimate_initial_beta(distances, perplexity);
    let mut betamin = f64::NEG_INFINITY;
    let mut betamax = f64::INFINITY;

    let mut p = vec![0.0; distances.len()];
    for _ in 0..PERPLEXITY_MAXITER {
        let res = compute_h(distances, beta);
        let h = res.0;
        p = res.1;
        let diff = h - log_perp;
        if diff.abs() < PERPLEXITY_ERROR {
            break;
        }
        if diff > 0.0 {
            betamin = beta;
            beta = if betamax.is_infinite() { beta * 2.0 } else { (beta + betamax) / 2.0 };
        } else {
            betamax = beta;
            beta = if betamin.is_infinite() { beta / 2.0 } else { (beta + betamin) / 2.0 };
        }
    }

    p
}

pub fn stochastic_outlier_selection<'a, S, D, F>(
    tree: &S, data: &'a D, perplexity: f64,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "SOS", false, F::zero(), F::zero(), F::infinity());
    }

    let mut scores: Vec<f64> = vec![1.0; size];

    let k = size.saturating_add(1);
    for idx in 0..size {
        let mut query = data.query();
        query.set_index(idx);
        let neighbors: Vec<(usize, F)> = tree
            .search_knn(&query, k)
            .into_iter()
            .filter(|neighbor| neighbor.index != idx)
            .map(|neighbor| (neighbor.index, neighbor.distance))
            .collect();

        let neighbors = if neighbors.len() > size.saturating_sub(1) {
            let mut truncated = neighbors;
            truncated.truncate(size.saturating_sub(1));
            truncated
        } else {
            neighbors
        };

        let distances: Vec<f64> =
            neighbors.iter().map(|(_, d)| d.to_f64().unwrap_or(f64::INFINITY)).collect();

        let p = compute_pi(&distances, perplexity);
        let sum_p: f64 = p.iter().sum();

        for ((neighbor_idx, _), &x) in neighbors.iter().zip(p.iter()) {
            if x > 0.0 && sum_p > 0.0 {
                let v = x / sum_p;
                scores[*neighbor_idx] *= 1.0 - v;
            }
        }
    }

    let scores: Vec<F> = scores.into_iter().map(|s| F::from_f64(s).unwrap_or(F::zero())).collect();

    make_outlier_result(scores, "SOS", false, F::zero(), F::zero(), F::infinity())
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
    fn sos_test() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = stochastic_outlier_selection(&tree, &data, 2.0);
        assert!(!results.scores.is_empty());
    }

    #[test]
    fn sos_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = stochastic_outlier_selection(&tree, &data, 4.5);
        println!("SOS first 10: {:?}", &result.scores[0..10]);
        let reference = load_reference_scores();
        let expected = reference.get("SOS-4.5").expect("No reference for SOS-4.5");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "SOS-4.5",
            auroc(&result.scores, &labels),
            auroc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("SOS-4.5", &result.scores, expected, 1e-6);
    }
}
