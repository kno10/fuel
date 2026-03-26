use crate::outlier::common::{OutlierResult, for_each_knn, make_outlier_result};
use crate::{DistanceData, Float, KnnSearch};

pub fn flexible_lof<'a, S, D, F>(
    tree: &S, data: &'a D, krefer: usize, kreach: usize,
) -> OutlierResult<F>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    let size = data.size();
    if size == 0 {
        return make_outlier_result(
            Vec::new(),
            "FlexibleLOF",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let krefer_effective = krefer.min(size.saturating_sub(1));
    let kreach_effective = kreach.min(size.saturating_sub(1));
    let k_effective = krefer_effective.max(kreach_effective);

    if k_effective == 0 {
        return make_outlier_result(
            vec![F::one(); size],
            "FlexibleLOF",
            false,
            F::zero(),
            F::zero(),
            F::infinity(),
        );
    }

    let neighborhoods = for_each_knn(tree, data, k_effective, false, |_, neigh| neigh);

    let neighbor_kreach_distance: Vec<F> = neighborhoods
        .iter()
        .map(|neigh| {
            if neigh.is_empty() {
                F::infinity()
            } else if neigh.len() >= kreach_effective {
                neigh[kreach_effective - 1].1
            } else {
                neigh.last().unwrap().1
            }
        })
        .collect();

    let mut lrd = vec![F::zero(); size];

    for i in 0..size {
        let neigh = &neighborhoods[i];
        if neigh.is_empty() {
            lrd[i] = F::infinity();
            continue;
        }

        let mut sum = F::zero();
        let mut count = F::zero();

        for (neighbor_idx, distance) in neigh.iter().take(kreach_effective) {
            let neighbor_kreach = neighbor_kreach_distance[*neighbor_idx];
            let reach_dist = if neighbor_kreach > *distance { neighbor_kreach } else { *distance };
            sum = sum + reach_dist;
            count = count + F::one();
        }

        lrd[i] = if sum > F::zero() { count / sum } else { F::infinity() };
    }

    let scores: Vec<F> = (0..size)
        .map(|i| {
            let neigh = &neighborhoods[i];

            if neigh.is_empty() || lrd[i].is_infinite() {
                return F::one();
            }

            let neighbors_ref = if neigh.len() > krefer_effective {
                &neigh[..krefer_effective]
            } else {
                &neigh[..]
            };

            let sum_neighbor_lrd: F = neighbors_ref.iter().map(|(idx, _)| lrd[*idx]).sum();
            let count = F::from_usize(neighbors_ref.len()).unwrap_or(F::zero());

            if lrd[i].is_infinite() || count == F::zero() {
                F::one()
            } else {
                sum_neighbor_lrd / (lrd[i] * count)
            }
        })
        .collect();

    make_outlier_result(scores, "FlexibleLOF", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;

    #[test]
    fn flexible_lof_remote_is_outlier() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![0.1, 0.1],
            vec![0.05, 0.05],
            vec![5.0, 5.0],
        ];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree: crate::vptree::VPTree<f64> =
            crate::vptree::VPTree::new(&data, 2, &mut rand::rngs::StdRng::seed_from_u64(0));

        let results = flexible_lof(&tree, &data, 2, 3);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, points.len() - 1);
    }
}
