use std::iter::Sum;
use std::ops::*;

use crate::cluster::kmeans::kgeometric::weiszfeld_step;
use crate::math;
use crate::outlier::common::{for_each_knn, make_outlier_result, OutlierResult};
use crate::{DistanceData, Float, KnnSearch, VectorData};

/// The method used to compute the central point of a neighborhood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenterType {
    /// Arithmetic mean: average of all points in the neighborhood.
    ArithmeticMean,
    /// Per-axis median: median coordinate for each axis independently across the neighborhood.
    PerAxisMedian,
    /// Geometric median: point minimizing the sum of L2 distances to points in the neighborhood.
    GeometricMedian,
}

/// Compute outlier scores based on the distance of each point to the center of its neighborhood.
///
/// Higher scores indicate stronger outlierness.
pub fn explainable_knn_outlier<'a, S, D, F>(
    tree: &S,
    data: &'a D,
    k: usize,
    center_type: CenterType,
    tol: F,
    store_centers: bool,
) -> Result<OutlierResult<F>, String>
where
    F: Float + AddAssign + SubAssign + MulAssign + Sum + Copy + std::fmt::Display,
    D: VectorData<F> + DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
{
    assert!(k > 0, "k must be greater than 0");

    let d = data.ncols();
    let k_effective = k.min(data.len().saturating_sub(1));

    let results: Vec<(F, Vec<F>)> = for_each_knn(tree, data, k_effective, false, |idx, neighbors| {
        if neighbors.is_empty() {
            return (F::zero(), vec![F::zero(); d]);
        }

        let center = match center_type {
            CenterType::ArithmeticMean => {
                let mut sum = vec![F::zero(); d];
                for (n_idx, _) in &neighbors {
                    let p = data.point(*n_idx);
                    math::add_assign(&mut sum, p, d);
                }
                math::scale(&mut sum, F::from(neighbors.len()).unwrap_or(F::one()).recip());
                sum
            }
            CenterType::PerAxisMedian => {
                let mut center = vec![F::zero(); d];
                for j in 0..d {
                    let mut coords: Vec<F> = neighbors.iter().map(|(n_idx, _)| data.point(*n_idx)[j]).collect();
                    coords.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let m = coords.len();
                    if m % 2 == 1 {
                        center[j] = coords[m / 2];
                    } else {
                        center[j] = (coords[m / 2 - 1] + coords[m / 2]) / F::from(2).unwrap_or(F::one());
                    }
                }
                center
            }
            CenterType::GeometricMedian => {
                // Initial guess: arithmetic mean
                let mut current = vec![F::zero(); d];
                for (n_idx, _) in &neighbors {
                    let p = data.point(*n_idx);
                    math::add_assign(&mut current, p, d);
                }
                math::scale(&mut current, F::from(neighbors.len()).unwrap_or(F::one()).recip());

                let mut assign = vec![1_usize; data.len()];
                for (n_idx, _) in &neighbors {
                    assign[*n_idx] = 0;
                }

                let max_iter = 100;
                let mut iter = 0;

                loop {
                    let next = weiszfeld_step::<F, _>(
                        data,
                        0,
                        &assign,
                        &current,
                        None,
                        false,
                    );

                    let dist_sq = math::sqdist(&next, &current, d);

                    current = next;
                    iter += 1;
                    if dist_sq.sqrt() <= tol || iter >= max_iter {
                        break;
                    }
                }
                current
            }
        };

        let q_point = data.point(idx);
        let score = match center_type {
            CenterType::PerAxisMedian => {
                math::l1dist(q_point, &center, d)
            }
            _ => {
                math::sqdist(q_point, &center, d).sqrt()
            }
        };

        (score, center)
    })?;

    let scores: Vec<F> = results.iter().map(|r| r.0).collect();
    let expected_positions = if store_centers {
        Some(results.into_iter().map(|r| r.1).collect())
    } else {
        None
    };

    Ok(make_outlier_result(
        scores,
        &format!("Explainable KNN Outlier ({:?}, k={})", center_type, k),
        false,
        F::zero(),
        F::zero(),
        F::infinity(),
        expected_positions,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NdArrayDataset;
    use ndarray::Array2;
    use crate::search::kdtree::KdTree;
    use crate::search::kdtree::MaxVarianceSplit;

    fn setup_data() -> (NdArrayDataset<f64>, KdTree<f64>) {
        let mat = Array2::from_shape_vec((5, 2), vec![
            0.0, 0.0,
            0.1, 0.0,
            0.0, 0.1,
            0.1, 0.1,
            6.0, 6.0,
        ]).unwrap();
        let dataset = NdArrayDataset::new(&mat);
        let tree = KdTree::new(&dataset, MaxVarianceSplit);
        (dataset, tree)
    }

    #[test]
    fn test_exknn_arithmetic() {
        let (data, tree) = setup_data();
        let res = explainable_knn_outlier(&tree, &data, 3, CenterType::ArithmeticMean, 1e-8, false).unwrap();
        assert!(res.scores[4] > res.scores[0]);
    }

    #[test]
    fn test_exknn_median() {
        let (data, tree) = setup_data();
        let res = explainable_knn_outlier(&tree, &data, 3, CenterType::PerAxisMedian, 1e-8, false).unwrap();
        assert!(res.scores[4] > res.scores[0]);
    }

    #[test]
    fn test_exknn_geometric() {
        let (data, tree) = setup_data();
        let res = explainable_knn_outlier(&tree, &data, 3, CenterType::GeometricMedian, 1e-8, false).unwrap();
        assert!(res.scores[4] > res.scores[0]);
    }
}
