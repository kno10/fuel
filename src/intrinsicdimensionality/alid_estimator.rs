use crate::api::IndexQuery;
use crate::intrinsicdimensionality::KNNIDEstimator;

/// ALID estimator of the intrinsic dimensionality (maximum likelihood estimator
/// for ID using auxiliary distances).
///
/// Reference:
///
/// Oussama Chelly, Michael E. Houle, Ken-ichi Kawarabayashi
/// Enhanced Estimation of Local Intrinsic Dimensionality Using Auxiliary Distances
/// Contributed to ELKI
///
/// This method uses a primary neighborhood radius r and for each neighbor i
/// computes contributions from neighbor j in the secondary neighborhood of i.
/// For each neighbor, it accumulates:
/// \(L = \ln(v/w)\) or \(\ln((v-w)/w)\) terms with cascading lookups.
/// Final estimate: \(\hat{m} = -a / sum\) where a is number of valid contributions.
///
/// Returns `NaN` on invalid distances or insufficient data.
pub fn alid<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
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
    if !w.is_finite() || w <= 0.0 {
        return f64::NAN;
    }

    let (halfw, mut a, mut sum) = (0.5 * w, 0, 0.0);
    for neighbor in neighbors.iter() {
        let v = neighbor.distance.to_f64().unwrap_or(f64::INFINITY);
        if !v.is_finite() || v <= 0.0 || neighbor.index == query_idx {
            continue;
        }
        sum += if v < halfw { (v / w).ln() } else { ((v - w) / w).ln_1p() };
        a += 1;

        let nw = w - v;
        if nw.is_nan() || nw <= 0.0 {
            continue;
        }
        let halfnw = 0.5 * nw;
        let inner_query = data.query().with_index(neighbor.index);
        for neighbor2 in tree.search_knn(&inner_query, k) {
            let v2 = neighbor2.distance.to_f64().unwrap_or(f64::INFINITY);
            if !v2.is_finite() || v2 <= 0.0 || neighbor2.index == neighbor.index || v2 > nw {
                continue;
            }
            let term2 = if v2 < halfnw { (v2 / nw).ln() } else { ((v2 - nw) / nw).ln_1p() };
            sum += term2;
            a += 1;
        }
    }

    if a == 0 || sum == 0.0 { f64::NAN } else { -((a as f64) / sum) }
}

pub struct ALID;

impl KNNIDEstimator for ALID {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        alid(tree, data, query_idx, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::make_intrinsic_subspace_data;
    use crate::search::kdtree::{AxisCycleSplit, KdTree};

    #[test]
    fn alid_estimator_smoke() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = TableWithDistance::with_distance(&data, Euclidean);
        let tree = KdTree::new(&table, AxisCycleSplit);
        let estimate = ALID::estimate_from_knn(&tree, &table, 0, 200);
        let expected = 5.399605402602233;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "ALID estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
