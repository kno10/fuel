use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

pub fn local_density_outlier_factor<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "LDOF", false, F::zero(), F::zero(), F::infinity());
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::one(); size],
            "LDOF",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);

    let scores: Vec<F> = (0..size)
        .par_map(|i| {
            let neigh = &neighborhoods[i];
            if neigh.is_empty() {
                return F::one();
            }

            let dxp: f64 = neigh.iter().map(|(_, d)| d.to_f64().unwrap_or(0.0)).sum::<f64>()
                / (neigh.len() as f64);

            let mut pair_sum = 0.0;
            let mut pair_count = 0;

            for (u_idx, _) in neigh.iter() {
                for (v_idx, _) in neigh.iter() {
                    if u_idx >= v_idx {
                        continue;
                    }
                    pair_sum += data.distance(*u_idx, *v_idx).to_f64().unwrap_or(0.0);
                    pair_count += 1;
                }
            }

            let d_xp = if pair_count > 0 { pair_sum / pair_count as f64 } else { 0.0 };

            let lf = if d_xp <= 0.0 { 1.0 } else { dxp / d_xp };
            let score = if lf.is_nan() || lf.is_infinite() { 1.0 } else { lf };

            F::from_f64(score).unwrap_or(F::zero())
        });

    make_outlier_result(scores, "LDOF", false, F::zero(), F::zero(), F::infinity())
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
    fn ldof_remote_outlier() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));
        let results = local_density_outlier_factor(&tree, &data, 2);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }

    #[test]
    fn ldof_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = local_density_outlier_factor(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("LDOF-10").expect("No reference for LDOF-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "LDOF-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("LDOF-10", &result.scores, expected, 1e-6);
    }
}
