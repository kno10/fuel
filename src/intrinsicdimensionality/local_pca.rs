use ndarray::Array2;
use ndarray_linalg::{Eigh, UPLO};

use crate::api::IndexQuery;
use crate::{DistanceData, Float, KnnSearch};

/// Local PCA intrinsic dimensionality estimator.
///
/// Reference:
/// - A. K. Jain, "Fundamentals of Digital Image Processing" (PCA as a dimension estimator)
///
/// Given k-nearest neighbor points, compute covariance matrix `C` and spectral decomposition:
/// \(C = V \Lambda V^T\).
///
/// ID is the smallest integer `m` such that:
/// \(\sum_{i=1}^m \lambda_i / \sum_{j=1}^d \lambda_j \ge \alpha\).
pub fn local_pca_id<'a, S, D, F>(
    tree: &S, data: &'a D, query_idx: usize, k: usize, alpha: f64,
) -> f64
where
    F: Float,
    D: DistanceData<F> + crate::VectorData<F> + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    if k < 2 || !(alpha > 0.0 && alpha <= 1.0) {
        return f64::NAN;
    }

    let query = data.query().with_index(query_idx);
    let neighbors: Vec<_> = tree
        .search_knn(&query, k + 1)
        .into_iter()
        .filter(|n| n.index != query_idx)
        .take(k)
        .collect();

    if neighbors.len() < k {
        return f64::NAN;
    }

    let dim = data.dims();
    if dim == 0 {
        return f64::NAN;
    }

    // Build centered data matrix of size (k x dim)
    let mut points = Vec::with_capacity(k * dim);
    for n in neighbors.iter() {
        let pt = data.point(n.index);
        for &x in pt.iter().take(dim) {
            points.push(x.to_f64().unwrap_or(0.0));
        }
    }

    let mut mean = vec![0.0; dim];
    for chunk in points.chunks_exact(dim) {
        for (j, &value) in chunk.iter().enumerate() {
            mean[j] += value;
        }
    }
    for m in mean.iter_mut() {
        *m /= k as f64;
    }

    for chunk in points.chunks_exact_mut(dim) {
        for (j, val) in chunk.iter_mut().enumerate() {
            *val -= mean[j];
        }
    }

    // Covariance matrix (d x d)
    let mut cov = vec![0.0; dim * dim];
    if k > 1 {
        let denom = (k - 1) as f64;
        for r in 0..dim {
            for c in 0..dim {
                let mut sum = 0.0;
                for i in 0..k {
                    sum += points[i * dim + r] * points[i * dim + c];
                }
                cov[r * dim + c] = sum / denom;
            }
        }
    } else {
        return f64::NAN;
    }

    let cov_matrix = match Array2::from_shape_vec((dim, dim), cov) {
        Ok(m) => m,
        Err(_) => return f64::NAN,
    };

    let eigvals = match cov_matrix.eigh(UPLO::Lower) {
        Ok((vals, _vecs)) => vals,
        Err(_) => return f64::NAN,
    };

    let mut variances: Vec<f64> =
        eigvals.iter().copied().filter(|x| x.is_finite() && *x > 0.0).collect();

    if variances.is_empty() {
        return f64::NAN;
    }

    variances.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let total_variance: f64 = variances.iter().sum();
    if total_variance.is_nan() || total_variance <= 0.0 {
        return f64::NAN;
    }

    let mut cum = 0.0;
    for (i, &v) in variances.iter().enumerate() {
        cum += v;
        if cum / total_variance >= alpha {
            return (i + 1) as f64;
        }
    }

    // If we didn't reach alpha because not enough components, return dimension.
    dim as f64
}

/// Type wrapper for Local PCA ID estimator.
pub struct LocalPCAID;

impl LocalPCAID {
    pub fn estimate_from_knn<'a, S, D, F>(
        tree: &S, data: &'a D, query_idx: usize, k: usize, alpha: f64,
    ) -> f64
    where
        F: Float,
        D: DistanceData<F> + crate::VectorData<F> + 'a,
        S: KnnSearch<F, D::Query<'a>> + Sync,
    {
        local_pca_id(tree, data, query_idx, k, alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TableWithDistance;
    use crate::distance::EuclideanDistance;
    use crate::intrinsicdimensionality::test::make_intrinsic_subspace_data;
    use crate::kd::{AxisCycleSplit, KdTree};

    #[test]
    fn local_pca_estimator_linspace() {
        let points =
            vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0], vec![4.0, 4.0]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree = KdTree::new(&data, AxisCycleSplit);

        let dim = LocalPCAID::estimate_from_knn(&tree, &data, 0, 4, 0.95);
        assert!((1.0..=2.0).contains(&dim));
    }

    #[test]
    fn local_pca_estimator_full_plane() {
        let points =
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let tree = KdTree::new(&data, AxisCycleSplit);

        let dim = LocalPCAID::estimate_from_knn(&tree, &data, 0, 4, 0.95);
        assert!((1.0..=2.0).contains(&dim));
    }

    #[test]
    fn local_pca_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = crate::data::TableWithDistance::with_distance(
            &data,
            crate::distance::EuclideanDistance,
        );
        let tree = crate::kd::KdTree::new(&table, crate::kd::AxisCycleSplit);

        let estimate = LocalPCAID::estimate_from_knn(&tree, &table, 0, 100, 0.95);
        let expected = 5.0;

        assert!(
            (estimate - expected).abs() < 1e-6,
            "local PCA estimate {} deviates from hypersphere expected {}",
            estimate,
            expected
        );
    }
}
