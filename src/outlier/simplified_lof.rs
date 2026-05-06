use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, ParMap};

pub fn simplified_lof<'a, S, D, F>(tree: &S, data: &'a D, k: usize) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "SimplifiedLOF",
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
            "SimplifiedLOF",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);

    let lrd: Vec<F> = (0..size).par_map(|i| {
        let neigh = &neighborhoods[i];
        if neigh.is_empty() {
            return F::infinity();
        }
        let sum: F = neigh.iter().map(|(_, d)| *d).sum();
        let count = F::from_usize(neigh.len()).unwrap_or(F::zero());
        if sum > F::zero() { count / sum } else { F::infinity() }
    });

    let scores: Vec<F> = (0..size)
        .par_map(|i| {
            let neigh = &neighborhoods[i];
            if neigh.is_empty() || lrd[i].is_infinite() {
                return F::one();
            }
            let sum_neighbors: F = neigh.iter().map(|(idx, _)| lrd[*idx]).sum();
            let count = F::from_usize(neigh.len()).unwrap_or(F::zero());
            if count > F::zero() { sum_neighbors / (lrd[i] * count) } else { F::one() }
        });

    make_outlier_result(scores, "SimplifiedLOF", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auc;
    use crate::outlier::common::*;

    #[test]
    fn simplified_lof_remote_is_outlier() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = simplified_lof(&tree, &data, 2);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }

    #[test]
    fn simplified_lof_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result = simplified_lof(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected =
            reference.get("SimplifiedLOF-10").expect("No reference for SimplifiedLOF-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "SimplifiedLOF-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("SimplifiedLOF-10", &result.scores, expected, 1e-6);
    }
}
