use crate::intrinsicdimensionality::KNNIDEstimator;
use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

pub fn local_intrinsic_dimensionality<'a, S, D, F, E>(
    tree: &S, data: &'a D, k: usize,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    E: KNNIDEstimator,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "LID", false, F::zero(), F::zero(), F::infinity());
    }

    let scores: Vec<F> = (0..size).par_map(|i| {
        let estimate = E::estimate_from_knn(tree, data, i, k + 1);
        if estimate.is_finite() && estimate > 0.0 { F::from_f64(estimate).unwrap_or(F::zero()) } else { F::zero() }
    });

    make_outlier_result(scores, "LID", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::intrinsicdimensionality::HillID;
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn lid_10_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = local_intrinsic_dimensionality::<
            _,
            _,
            _,
            crate::intrinsicdimensionality::MethodOfMoments,
        >(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("LID-10").expect("No reference for LID-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "LID-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            0.2,
        );
        assert_outlier_scores_approx("LID-10", &result.scores, expected, 10.0);
    }

    #[test]
    fn lid_20_hill_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = local_intrinsic_dimensionality::<_, _, _, HillID>(&tree, &data, 20);
        let reference = load_reference_scores();
        let expected = reference.get("LID-20-Hill").expect("No reference for LID-20-Hill");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "LID-20-Hill",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-6,
        );
        assert_outlier_scores_approx("LID-20-Hill", &result.scores, expected, 1e-6);
    }
}
