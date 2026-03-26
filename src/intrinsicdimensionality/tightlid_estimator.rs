#![allow(non_snake_case)]
use crate::api::{DistanceData, IndexQuery};
use crate::intrinsicdimensionality::KNNIDEstimator;
use crate::{Float, KnnSearch};

/// TightLID Estimator (TLE) of the intrinsic dimensionality (maximum likelihood
/// estimator for ID using auxiliary distances).
///
/// Reference:
///
/// Laurent Amsaleg, Oussama Chelly, Michael E. Houle, Ken-ichi Kawarabayashi,
/// Milos Radovanovic, Weeris Treeratanajaru
/// Intrinsic Dimensionality Estimation within Tight Localities
/// Proc. 2019 SIAM International Conference on Data Mining (SDM)
///
/// Input:
/// - query radius or kNN radius \(r\)
/// - neighbor distances \(d_i\), auxiliary distances \(v_{ij}\)
///
/// Local terms:
/// \(D_i^2 = d_i^2\), \(r^2 - D_i^2 = R_i\)
/// \(S_{ij} = (\sqrt{(D_i^2 + V_{ij}^2 - D_j^2)^2 + 2 V_{ij}^2 R_i} - (D_i^2 + V_{ij}^2 - D_j^2)) / (2 R_i)\)
/// and similarly for \(T_{ij}\) with \(Z_{ij}=2D_i^2+2D_j^2-V_{ij}^2\).
///
/// Aggregate it as
/// \(\sum \ln(T_{ij}) + \ln(S_{ij}) + \ln(D_i^2 / r^2)\)
/// with count \(N\), returning \(m = -N / \sum(... )\) when sum < 0.
///
/// Returns `NaN` if `k < 2`, insufficient neighbors, or invalid distances.
pub fn tightlid<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
where
    F: Float,
    D: DistanceData<F> + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    if k < 2 {
        return f64::NAN;
    }

    let query = data.query().with_index(query_idx);
    let knn: Vec<_> = tree
        .search_knn(&query, k + 1)
        .into_iter()
        .filter(|n| n.index != query_idx)
        .take(k)
        .collect();

    let n = knn.len();
    if n < 2 {
        return f64::NAN;
    }

    let r =
        knn.last().map(|n| n.distance.to_f64().unwrap_or(f64::INFINITY)).unwrap_or(f64::INFINITY);
    if r.is_nan() || r <= 0.0 {
        return f64::NAN;
    }

    let r2 = r * r;
    let (mut sum, mut valid) = (0.0, 0usize);
    for i in 0..n {
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

        for j in (i + 1)..n {
            let kdj = knn[j].distance.to_f64().unwrap_or(f64::NAN);
            if kdj.is_nan() || kdj <= 0.0 || kdj >= r {
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
            s = (s - (di2 + v2 - dj2)) * ir2mDi2;
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

pub struct TightLID;

impl KNNIDEstimator for TightLID {
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: Float,
        D: DistanceData<F> + 'a,
        S: KnnSearch<F, D::Query<'a>> + Sync,
    {
        tightlid(tree, data, query_idx, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::intrinsicdimensionality::test::make_intrinsic_subspace_data;
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
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let tree = KdTree::new(&data, LargestSpreadSplit);

        let est = TightLID::estimate_from_knn(&tree, &data, 0, 4);
        assert!(est.is_finite());
        assert!(est > 0.0);
    }

    #[test]
    fn tightlid_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(10000, 0);
        let table = TableWithDistance::with_distance(&data, Euclidean);
        let tree = KdTree::new(&table, LargestSpreadSplit);

        let est = TightLID::estimate_from_knn(&tree, &table, 0, 200);
        let expected = 4.980667540371235;
        assert!(
            (est - expected).abs() < 1e-6,
            "tightlid estimate {} deviates from data-based expected {}",
            est,
            expected
        );
    }
}
