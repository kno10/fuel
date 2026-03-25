use rs_stats::utils::special_functions::gamma_fn;

use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, VectorData};

pub fn variance_of_volume<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.size();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "VarianceOfVolume",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }
    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::one(); size],
            "VarianceOfVolume",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);

    let dim = data.dims() as f64;
    let scale_const = std::f64::consts::PI.sqrt() * gamma_fn(1.0 + dim * 0.5).powf(-1.0 / dim);

    let mut volumes = vec![0.0; size];

    for i in 0..size {
        let neigh = &neighborhoods[i];
        let dist_k =
            if neigh.is_empty() { 0.0 } else { neigh.last().unwrap().1.to_f64().unwrap_or(0.0) };
        let vol = if dist_k > 0.0 { (dist_k * scale_const).powf(dim) } else { 0.0 };
        volumes[i] = vol;
    }

    let sum_vol: f64 = volumes.iter().sum();
    let scaling = if sum_vol > 0.0 { (size as f64) / sum_vol } else { 1.0 };

    let scaled_volumes: Vec<f64> = volumes.iter().map(|v| v * scaling).collect();

    let scores: Vec<F> = (0..size)
        .map(|i| {
            let neigh = &neighborhoods[i];
            let total_neighbors = neigh.len() + 1;

            let mut vbar = scaled_volumes[i];
            for (idx, _) in neigh.iter() {
                vbar += scaled_volumes[*idx];
            }
            vbar /= total_neighbors as f64;

            let mut vov_sum = (scaled_volumes[i] - vbar).powi(2);
            for (idx, _) in neigh.iter() {
                vov_sum += (scaled_volumes[*idx] - vbar).powi(2);
            }

            let vov = if vov_sum.is_finite() {
                vov_sum / ((total_neighbors - 1) as f64)
            } else {
                f64::INFINITY
            };

            let score = if vov.is_nan() || vov.is_infinite() { 1.0 } else { vov };

            F::from_f64(score).unwrap_or(F::zero())
        })
        .collect();

    make_outlier_result(scores, "VarianceOfVolume", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::EuclideanDistance;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;

    #[test]
    fn variance_of_volume_remote_outlier() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree: crate::vptree::VPTree<f64> =
            crate::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));
        let results = variance_of_volume(&tree, &data, 2);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }

    #[test]
    fn variance_of_volume_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::vptree::VPTree<f64> = crate::vptree::VPTree::new(&data, 2, &mut rng);

        let result = variance_of_volume(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("VOV-10").expect("No reference for VOV-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "VOV-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("VOV-10", &result.scores, expected, 1e-6);
    }
}
