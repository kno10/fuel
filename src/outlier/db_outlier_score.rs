use crate::outlier::common::{OutlierResult, for_each_range, make_outlier_result};
use crate::{DistanceData, Float, RangeSearch};

pub fn db_outlier_score<'a, S, D, F>(tree: &S, data: &'a D, d: F) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: RangeSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    let scores: Vec<F> = for_each_range(tree, data, d, true, |_idx, neighbors| {
        let count = neighbors.len();
        let n = (count as f64) / (size as f64);
        let score_val = 1.0 - n;
        F::from_f64(score_val).unwrap_or(F::zero())
    });

    make_outlier_result(scores, "DBOutlierScore", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;

    #[test]
    fn db_outlier_score_test() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));
        let results = db_outlier_score(&tree, &data, 0.2);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, points.len() - 1);
    }

    #[test]
    fn db_outlier_score_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree = crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = db_outlier_score(&tree, &data, 0.25);
        let reference = load_reference_scores();
        let expected = reference.get("DBOS-10").expect("No reference for DBOS-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "DBOS-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("DBOS-10", &result.scores, expected, 1e-6);
    }
}
