use super::abod::abod_kernel_score_for_neighbor_set;
use crate::outlier::common::{OutlierResult, for_each_knn};
use crate::{DistanceData, Float, KnnSearch, ParMap, VectorData};

/// Fast-ABOD (approximate ABOD) using `k` nearest Euclidean neighbors and a kernel similarity function.
pub fn fast_angle_based_outlier_detection<'a, S, D, F, K>(
    tree: &S, data: &'a D, k: usize, kernel: K,
) -> Result<OutlierResult<F>, String>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    K: Fn(&[F], &[F]) -> F + Sync,
{
    let size = data.len();
    if size == 0 {
        return Ok(OutlierResult::builder(Vec::new(), "FastABOD")
            .theoretical_min(F::zero())
            .theoretical_max(F::infinity())
            .build());
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective < 2 {
        return Ok(OutlierResult::builder(vec![F::infinity(); size], "FastABOD")
            .theoretical_min(F::zero())
            .theoretical_max(F::infinity())
            .build());
    }

    let neighborhoods: Vec<Vec<(usize, F)>> =
        for_each_knn(tree, data, k_effective, false, |_idx, neighbors| neighbors)?;

    let scores: Vec<F> = (0..size).par_map(|i| {
        let neighbor_ids: Vec<usize> = neighborhoods[i].iter().map(|(idx, _)| *idx).collect();
        abod_kernel_score_for_neighbor_set(data, i, &neighbor_ids, &kernel)
    });

    Ok(OutlierResult::builder(scores, "FastABOD")
        .theoretical_min(F::zero())
        .theoretical_max(F::infinity())
        .build())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::outlier::common::*;
    use crate::search::vptree::VPTree;

    #[test]
    fn fast_abod_poly2_smoke() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let kernel = crate::kernel::polynomial::PolynomialKernel::new(2, 1.0, 0.0);
        let result =
            fast_angle_based_outlier_detection(&tree, &data, 10, |x, y| kernel.similarity(x, y))
                .unwrap();

        assert_eq!(result.scores.len(), points.len());
        assert!(result.scores.iter().all(|s| s.is_finite()));
    }

    #[test]
    fn fast_and_lbabod_poly2_smoke() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let kernel = crate::kernel::polynomial::PolynomialKernel::new(2, 1.0, 0.0);
        let fast =
            fast_angle_based_outlier_detection(&tree, &data, 10, |x, y| kernel.similarity(x, y))
                .unwrap();
        let result_lba =
            crate::outlier::lb_abod_kernel(&data, 10, 10, |x, y| kernel.similarity(x, y));

        assert_eq!(fast.scores.len(), points.len());
        assert!(fast.scores.iter().all(|s| s.is_finite()));
        assert_eq!(result_lba.scores.len(), points.len());
        assert!(result_lba.scores.iter().all(|s| s.is_finite()));
    }
}
