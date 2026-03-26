use super::abod::{abod_kernel_score_for_neighbor_set, abod_score_for_neighbor_set};
use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch, VectorData};

/// Fast-ABOD (approximate ABOD) using `k` nearest neighbors.
pub fn fast_angle_based_outlier_detection<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize,
) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.size();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "FastABOD",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective < 2 {
        return make_outlier_result(
            vec![F::infinity(); size],
            "FastABOD",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods: Vec<Vec<(usize, F)>> =
        for_each_knn(tree, data, k_effective, false, |_idx, neighbors| neighbors);

    let scores: Vec<F> = (0..size)
        .map(|i| {
            let neighbor_ids: Vec<usize> = neighborhoods[i].iter().map(|(idx, _)| *idx).collect();
            abod_score_for_neighbor_set(data, i, &neighbor_ids)
        })
        .collect();

    make_outlier_result(scores, "FastABOD", false, F::zero(), F::zero(), F::infinity())
}

/// Fast ABOD with a kernel similarity function (e.g., polynomial kernel).
pub fn fast_angle_based_outlier_detection_kernel<D, F, K>(
    data: &D, k: usize, kernel: K,
) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + VectorData<F> + Sync,
    K: Fn(&[F], &[F]) -> F + Sync,
{
    let size = data.size();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "FastABOD",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective < 2 {
        return make_outlier_result(
            vec![F::infinity(); size],
            "FastABOD",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let mut neighborhoods: Vec<Vec<usize>> = Vec::with_capacity(size);
    for i in 0..size {
        let xi = data.point(i);
        let sim_ii = kernel(xi, xi);
        let mut dist_idx: Vec<(F, usize)> = Vec::with_capacity(size - 1);

        for j in 0..size {
            if i == j {
                continue;
            }
            let xj = data.point(j);
            let sim_jj = kernel(xj, xj);
            let sim_ij = kernel(xi, xj);
            let sqd = sim_ii + sim_jj - sim_ij - sim_ij;
            dist_idx.push((sqd, j));
        }
        dist_idx.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        neighborhoods.push(dist_idx.into_iter().take(k_effective).map(|(_, idx)| idx).collect());
    }

    let scores: Vec<F> = (0..size)
        .map(|i| abod_kernel_score_for_neighbor_set(data, i, &neighborhoods[i], &kernel))
        .collect();

    make_outlier_result(scores, "FastABOD", false, F::zero(), F::zero(), F::infinity())
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
    use crate::vptree::VPTree;

    #[test]
    fn fast_abod_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree: VPTree<f64> = VPTree::new(&data, 2, &mut rng);

        let result = fast_angle_based_outlier_detection(&tree, &data, 10);
        let reference = load_reference_scores();
        let expected = reference.get("FastABOD-10").expect("No reference for FastABOD-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "FastABOD-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("FastABOD-10", &result.scores, expected, 1e-5);
    }

    #[test]
    fn fast_and_lbabod_poly2_smoke() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);

        let kernel = crate::kernel::polynomial::PolynomialKernel::new(2, 1.0, 0.0);
        let _fast =
            fast_angle_based_outlier_detection_kernel(&data, 10, |x, y| kernel.similarity(x, y));
        let result_lba = crate::outlier::locality_based_abod_kernel(&data, 10, 10, |x, y| {
            kernel.similarity(x, y)
        });

        assert_eq!(result_lba.scores.len(), points.len());
        assert!(result_lba.scores.iter().all(|s| s.is_finite()));
    }
}
