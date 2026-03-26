use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::outlier::kernel::KernelDensityFunction;
use crate::{DistanceData, Float, KnnSearch, VectorData};

pub fn simple_kernel_density_lof<'a, S, D, F>(
    tree: &S, data: &'a D, k: usize, _h: f64, kernel: KernelDensityFunction,
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
            "SimpleKernelDensityLOF",
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
            "SimpleKernelDensityLOF",
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
        .map(|neigh| neigh.last().map_or(0.0, |(_, d)| d.to_f64().unwrap_or(0.0)))
        .collect();

    let mut density = vec![0.0; size];

    for i in 0..size {
        let neigh = &neighborhoods[i];
        if neigh.is_empty() {
            density[i] = 0.0;
            continue;
        }

        let mut sum = 0.0;
        for (neighbor_idx, d) in neigh.iter() {
            let max_dist = kdistances[*neighbor_idx];
            if max_dist <= 0.0 {
                sum = f64::INFINITY;
                break;
            }
            let dist = d.to_f64().unwrap_or(0.0);
            let v = dist / max_dist;
            sum += kernel.density(v) / max_dist.powf(dim);
        }

        density[i] = if !neigh.is_empty() { sum / (neigh.len() as f64) } else { 0.0 };
    }

    let scores: Vec<F> = (0..size)
        .map(|i| {
            let own = density[i];
            let neigh = &neighborhoods[i];
            let sum_neighbors: f64 = neigh.iter().map(|(idx, _)| density[*idx]).sum();
            let mean_neighbors =
                if !neigh.is_empty() { sum_neighbors / (neigh.len() as f64) } else { 0.0 };

            let score = if own.is_nan() || own <= 0.0 {
                1.0
            } else if own.is_infinite() {
                if mean_neighbors.is_infinite() { 1.0 } else { 0.0 }
            } else {
                (mean_neighbors / own).max(0.0)
            };

            F::from_f64(score).unwrap_or(F::zero())
        })
        .collect();

    make_outlier_result(
        scores,
        "SimpleKernelDensityLOF",
        false,
        F::zero(),
        F::zero(),
        F::infinity(),
    )
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;

    #[test]
    fn simple_kernel_density_lof_remote_outlier() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::vptree::VPTree<f64> =
            crate::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));
        let results =
            simple_kernel_density_lof(&tree, &data, 2, 1.0, KernelDensityFunction::Epanechnikov);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }
}
