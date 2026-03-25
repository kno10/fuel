use crate::api::IndexQuery;
use crate::intrinsicdimensionality::KnnBasedIntrinsicDimensionalityEstimator;
pub struct ALIDEstimator;

impl KnnBasedIntrinsicDimensionalityEstimator for ALIDEstimator {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        let query = data.query().with_index(query_idx);
        let neighbors: Vec<_> = tree.search_knn(&query, k).into_iter().collect();

        if neighbors.is_empty() {
            return f64::NAN;
        }

        let w = neighbors
            .last()
            .map(|n| n.distance.to_f64().unwrap_or(f64::INFINITY))
            .unwrap_or(f64::INFINITY);
        if w.is_nan() || w <= 0.0 {
            return f64::NAN;
        }

        let halfw = 0.5 * w;
        let mut a = 0;
        let mut sum = 0.0;

        for neighbor in neighbors.iter() {
            let v = neighbor.distance.to_f64().unwrap_or(f64::INFINITY);
            if v.is_nan() || v <= 0.0 || neighbor.index == query_idx {
                continue;
            }
            let term = if v < halfw { (v / w).ln() } else { ((v - w) / w).ln_1p() };
            sum += term;
            a += 1;

            let nw = w - v;
            if nw.is_nan() || nw <= 0.0 {
                continue;
            }
            let halfnw = 0.5 * nw;
            let inner_query = data.query().with_index(neighbor.index);
            for neighbor2 in tree.search_knn(&inner_query, k) {
                let v2 = neighbor2.distance.to_f64().unwrap_or(f64::INFINITY);
                if v2.is_nan() || v2 <= 0.0 || neighbor2.index == neighbor.index || v2 > nw {
                    continue;
                }
                let term2 = if v2 < halfnw { (v2 / nw).ln() } else { ((v2 - nw) / nw).ln_1p() };
                sum += term2;
                a += 1;
            }
        }

        if a == 0 || sum == 0.0 {
            return f64::NAN;
        }
        -((a as f64) / sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TableWithDistance;
    use crate::distance::EuclideanDistance;
    use crate::intrinsicdimensionality::make_hypersphere_embedded_data;

    #[test]
    fn alid_estimator_smoke() {
        let data = make_hypersphere_embedded_data(1000, 0);
        let table = TableWithDistance::with_distance(&data, EuclideanDistance);
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);
        let estimate = ALIDEstimator::estimate_from_knn(&tree, &table, 0, 200);
        let expected = 4.706594375835201;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "ALID estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
