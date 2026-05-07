use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::outlier::kernel::KernelDensityFunction;
use crate::{DistanceData, Float, KnnSearch, ParMap, VectorData};

pub fn local_density_factor<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize, h: f64, c: f64, kernel: KernelDensityFunction,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.len();
    if size == 0 {
        return make_outlier_result(Vec::new(), "LDF", false, F::zero(), F::zero(), F::infinity());
    }

    let k_effective = k.min(size.saturating_sub(1));
    if k_effective == 0 {
        return make_outlier_result(
            vec![F::one(); size],
            "LDF",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);

    let dim = data.dims() as f64;

    let kdistances: Vec<f64> = neighborhoods
        .iter()
        .map(|neigh| {
            if neigh.is_empty() {
                f64::INFINITY
            } else {
                neigh.last().map_or(0.0, |(_, d)| d.to_f64().unwrap_or(0.0))
            }
        })
        .collect();

    let lde: Vec<f64> = (0..size).par_map(|i| {
        let neigh = &neighborhoods[i];
        if neigh.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        let mut is_inf = false;
        for (neighbor_idx, d) in neigh.iter() {
            let nkdist = kdistances[*neighbor_idx];
            let dij = d.to_f64().unwrap_or(0.0);
            if nkdist.is_nan() || nkdist <= 0.0 || nkdist.is_infinite() {
                is_inf = true;
                break;
            }
            let r = nkdist.max(dij);
            let factors = h * nkdist;
            let v = r / factors;
            let density_val = kernel.density(v);
            let term = density_val / factors.powf(dim);
            sum += term;
        }
        if is_inf { f64::INFINITY } else { sum / (neigh.len() as f64) }
    });

    let scores: Vec<F> = (0..size)
        .par_map(|i| {
            let own = lde[i];
            let neigh = &neighborhoods[i];
            let sum_neighbors: f64 = neigh.iter().map(|(idx, _)| lde[*idx]).sum();

            let mean_neighbors =
                if !neigh.is_empty() { sum_neighbors / (neigh.len() as f64) } else { 0.0 };

            let div = if own.is_infinite() { f64::INFINITY } else { own + c * mean_neighbors };

            let score = if div.is_infinite() {
                if mean_neighbors.is_infinite() { 1.0 } else { 0.0 }
            } else if div > 0.0 {
                if neigh.is_empty() { 1.0 } else { mean_neighbors / div }
            } else {
                0.0
            };

            F::from_f64(score).unwrap_or(F::zero())
        });

    make_outlier_result(scores, "LDF", false, F::zero(), F::zero(), F::infinity())
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
    fn ldf_remote_outlier() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));
        let results =
            local_density_factor(&tree, &data, 2, 1.0, 0.1, KernelDensityFunction::Gaussian);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }

    #[test]
    fn ldf_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let tree: crate::search::vptree::VPTree<f64> =
            crate::search::vptree::VPTree::new(&data, 2, &mut rng);

        let result =
            local_density_factor(&tree, &data, 10, 1.0, 0.1, KernelDensityFunction::Gaussian);
        let reference = load_reference_scores();
        let expected = reference.get("LDF-10").expect("No reference for LDF-10");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "LDF-10",
            auc(&result.scores, &labels),
            auc(expected, &labels),
            1e-12,
        );
        assert_outlier_scores_approx("LDF-10", &result.scores, expected, 1e-6);
    }
}
