use std::f64;

use crate::intrinsicdimensionality::KNNIDEstimator;
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch};

/// Intrinsic Dimensionality Outlier Score (IDOS).
///
/// This function uses an intrinsic dimensionality estimator on a context set of
/// `k_c` neighbors, then scores each point by with a reference set of `k_r`.
pub fn intrinsic_dimensionality_outlier_score<'a, S, D, F, E>(
    tree: &S, data: &'a D, k_c: usize, k_r: usize,
) -> OutlierResult<f64>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    E: KNNIDEstimator,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "IDOS", false, 0.0, 0.0, f64::INFINITY);
    }

    let mut ids = Vec::with_capacity(size);
    for i in 0..size {
        // following ELKI semantics (query+neighbors): use k_c + 1 here.
        let estimate = E::estimate_from_knn(tree, data, i, k_c + 1);
        let value = if estimate.is_finite() && estimate > 0.0 { estimate } else { 0.0 };
        ids.push(value);
    }

    let scores = crate::outlier::common::for_each_knn(tree, data, k_r - 1, false, |i, neigh| {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for &(neighbor_index, _) in &neigh {
            let neighbor_id = ids[neighbor_index];
            if neighbor_id > 0.0 {
                sum += 1.0 / neighbor_id;
            }
            cnt += 1;
        }

        let id_q = ids[i];
        if id_q > 0.0 && cnt > 0 { id_q * sum / (cnt as f64) } else { 0.0 }
    });

    make_outlier_result(scores, "IDOS", false, 0.0, 0.0, f64::INFINITY)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::intrinsicdimensionality::{AggregatedHillID, HillID};
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn idos_10_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = intrinsic_dimensionality_outlier_score::<_, _, _, AggregatedHillID>(
            &tree, &data, 10, 10,
        );
        let reference = load_reference_scores();
        let expected = reference.get("IDOS-10").expect("No reference for IDOS-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        let auc_res = auc(&result.scores, &labels);
        let auc_exp = auc(expected, &labels);
        eprintln!("IDOS-10: auc_result={} auc_expected={}", auc_res, auc_exp);
        eprintln!("IDOS-10 first 20 res: {:?}", &result.scores[..20]);
        eprintln!("IDOS-10 first 20 exp: {:?}", &expected[..20]);

        assert_outlier_auc_approx("IDOS-10", auc_res, auc_exp, 1e-6);
        assert_outlier_scores_approx("IDOS-10", &result.scores, expected, 1e-6);
    }

    #[test]
    fn idos_20_hill_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result =
            intrinsic_dimensionality_outlier_score::<_, _, _, HillID>(&tree, &data, 20, 20);
        let reference = load_reference_scores();
        let expected = reference.get("IDOS-20-Hill").expect("No reference for IDOS-20-Hill");
        let labels: Vec<u8> = label_from_reference(&reference);

        let auc_res = auc(&result.scores, &labels);
        let auc_exp = auc(expected, &labels);
        println!("IDOS-20-Hill: auc_result={} auc_expected={}", auc_res, auc_exp);
        println!("IDOS first 20 res: {:?}", &result.scores[..20]);
        println!("IDOS first 20 exp: {:?}", &expected[..20]);
        assert_outlier_auc_approx("IDOS-20-Hill", auc_res, auc_exp, 1e-6);
        assert_outlier_scores_approx("IDOS-20-Hill", &result.scores, expected, 1e-6);
    }
}
