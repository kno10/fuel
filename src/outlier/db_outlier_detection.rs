use crate::outlier::common::{OutlierResult, for_each_range, make_outlier_result};
use crate::{DistanceData, Float, RangeSearch};

pub fn db_outlier_detection<'a, S, D, F>(tree: &S, data: &'a D, d: F, p: f64) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: RangeSearch<F, D::Query<'a>> + Sync,
{
    let size = data.size();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "DBOutlierDetection",
            false,
            F::one(),
            F::zero(),
            F::infinity(),
        );
    }

    let m = ((size as f64) * (1.0 - p)).floor() as usize;

    let scores: Vec<F> = for_each_range(tree, data, d, true, |_idx, neighbors| {
        let count = neighbors.len();
        let outlier = if count < m { 1.0 } else { 0.0 };
        F::from_f64(outlier).unwrap_or(F::zero())
    });

    make_outlier_result(scores, "DBOutlierDetection", false, F::zero(), F::zero(), F::infinity())
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
    fn db_outlier_detection_test() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
        ];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let tree = crate::vptree::VPTree::new(&data, 2, &mut rng);
        let results = db_outlier_detection(&tree, &data, 0.2, 0.2);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, points.len() - 1);
    }

    #[test]
    fn db_outlier_detection_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree = crate::vptree::VPTree::new(&data, 2, &mut rng);

        let result = db_outlier_detection(&tree, &data, 0.25, 0.95);
        let reference = load_reference_scores();
        let expected = reference.get("DBOD-10").expect("No reference for DBOD-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "DBOD-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("DBOD-10", &result.scores, expected, 1e-6);
    }

    #[test]
    fn db_outlier_detection_counts_self_behavior() {
        let points = vec![vec![0.0], vec![0.1], vec![100.0]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree = crate::vptree::VPTree::new(&data, 2, &mut rng);

        // p=0.1 => m = floor(3*(1-0.1)) = 2
        // point 0: count=2 (self+point1) -> inlier
        // point 1: count=2 -> inlier
        // point 2: count=1 -> outlier
        let result: crate::outlier::common::OutlierResult<f64> =
            db_outlier_detection(&tree, &data, 0.5, 0.1);
        assert_eq!(result.scores, vec![0.0, 0.0, 1.0]);
    }
}
