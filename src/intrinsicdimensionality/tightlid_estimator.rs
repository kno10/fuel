#![allow(non_snake_case)]
use crate::api::{DistanceData, IndexQuery};
use crate::intrinsicdimensionality::KnnBasedIntrinsicDimensionalityEstimator;
use crate::{Float, KnnSearch};

pub struct TightLIDEstimator;

pub fn tightlid_estimate_from_knn<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize,
) -> f64
where
    F: Float,
    D: DistanceData<F> + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    TightLIDEstimator::estimate_from_knn(tree, data, query_idx, k)
}

impl KnnBasedIntrinsicDimensionalityEstimator for TightLIDEstimator {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: Float,
        D: DistanceData<F> + 'a,
        S: KnnSearch<F, D::Query<'a>> + Sync,
    {
        if k < 2 {
            return f64::NAN;
        }

        let query = data.query().with_index(query_idx);
        let mut knn: Vec<_> = tree
            .search_knn(&query, k + 1)
            .into_iter()
            .filter(|n| n.index != query_idx)
            .take(k)
            .collect();

        if knn.len() < k {
            return f64::NAN;
        }

        // Ensure sorted by distance ascending.
        knn.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });

        let r = knn
            .last()
            .map(|n| n.distance.to_f64().unwrap_or(f64::INFINITY))
            .unwrap_or(f64::INFINITY);
        if r.is_nan() || r <= 0.0 {
            return f64::NAN;
        }

        let r2 = r * r;
        let mut sum = 0.0;
        let mut valid = 0usize;

        for i in 0..k {
            let kdi = knn[i].distance.to_f64().unwrap_or(f64::NAN);
            if kdi.is_nan() || kdi <= 0.0 || kdi >= r {
                continue;
            }
            let di2 = kdi * kdi;
            let r2mDi2 = 2.0 * (r2 - di2);
            if r2mDi2.is_nan() || r2mDi2 <= 0.0 {
                continue;
            }
            let ir2mDi2 = 1.0 / r2mDi2;

            for j in (i + 1)..k {
                let kdj = knn[j].distance.to_f64().unwrap_or(f64::NAN);
                if !(kdj > 0.0 && kdj < r) {
                    continue;
                }
                let dj2 = kdj * kdj;
                let dij = data.distance(knn[i].index, knn[j].index).to_f64().unwrap_or(f64::NAN);
                if dij.is_nan() || dij <= 0.0 {
                    continue;
                }
                let v2 = dij * dij;

                let mut s = di2 + v2 - dj2;
                s = (s * s + 2.0 * v2 * r2mDi2).sqrt();
                s = (s - (di2 + v2 - dj2)) * ir2mDi2; // matches Java expression
                if s.is_nan() || s <= 0.0 {
                    continue;
                }

                let z2 = 2.0 * di2 + 2.0 * dj2 - v2;
                let mut t = di2 + z2 - dj2;
                t = (t * t + 2.0 * z2 * r2mDi2).sqrt();
                t = (t - (di2 + z2 - dj2)) * ir2mDi2;

                if t > 0.0 {
                    sum += 2.0 * ((t.ln()) + (s.ln()));
                    valid += 2;
                }
            }

            sum += (di2 / r2).ln();
            valid += 1;
        }

        if sum < 0.0 { -((valid as f64) / sum) } else { 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TableWithDistance;
    use crate::distance::EuclideanDistance;
    use crate::kd::{KdTree, LargestSpreadSplit};

    #[test]
    fn tightlid_estimator_basic() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 1.0],
        ];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree = KdTree::new(&data, LargestSpreadSplit);

        let est = TightLIDEstimator::estimate_from_knn(&tree, &data, 0, 4);
        assert!(est.is_finite());
        assert!(est > 0.0);
    }

    #[test]
    fn tightlid_estimator_not_enough_neighbors() {
        let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree = KdTree::new(&data, LargestSpreadSplit);

        let est = TightLIDEstimator::estimate_from_knn(&tree, &data, 0, 5);
        assert!(est.is_nan());
    }

    #[test]
    fn tightlid_estimator_hypersphere_close_to_5() {
        let data = crate::intrinsicdimensionality::make_hypersphere_embedded_data(10000, 0);
        let table = TableWithDistance::with_distance(&data, EuclideanDistance);
        let tree = KdTree::new(&table, LargestSpreadSplit);

        let est = TightLIDEstimator::estimate_from_knn(&tree, &table, 0, 200);
        let expected = 4.435060777302711;
        assert!(
            (est - expected).abs() < 1e-6,
            "tightlid estimate {} deviates from data-based expected {}",
            est,
            expected
        );
    }
}
