use crate::Float;
use crate::intrinsicdimensionality::{DistanceIDEstimator, find_begin};

/// Zipf estimator (qq-estimator) of intrinsic dimensionality.
///
/// Reference:
///
/// M. Kratz, S. I. Resnick
/// The QQ-estimator and heavy tails
/// Communications in Statistics. Stochastic Models 12(4)
///
/// J. Schultze, J. Steinebach
/// On Least Squares Estimates of an Exponential Tail Coefficient
/// Statistics & Risk Modeling 14(4)
///
/// J. Beirlant, G. Dierckx, A. Guillou
/// Estimation of the extreme-value index and generalized quantile plots
/// Bernoulli 11(6)
///
/// Uses weighted least-squares regression on \(\ln(r_i)\) vs \(\ln(d_i)\) with bias-corrected rank weights.
///
/// Returns `NaN` for insufficient or invalid data.
pub fn zipf_id<F: Float>(distances: &[F]) -> f64 {
    let begin = find_begin(distances);
    let len_d = distances.len() - begin;
    if len_d < 2 {
        return f64::NAN;
    }
    let len = len_d as f64;
    let bias = 0.6;
    let nplus1 = len + bias;

    let (mut wls, mut ws, mut ls, mut wws) = (0.0, 0.0, 0.0, 0.0);
    for (i, &v) in distances[begin..].iter().enumerate() {
        let v64 = crate::intrinsicdimensionality::positive_f64(v);
        if v64.is_nan() {
            continue;
        }
        let logv = v64.ln();
        let weight = (nplus1 / ((i as f64) + bias)).ln();
        wls += weight * logv;
        ws += weight;
        ls += logv;
        wws += weight * weight;
    }

    let denom = len * wws - ws * ws;
    if denom == 0.0 { f64::NAN } else { -1.0 / ((len * wls - ws * ls) / denom) }
}

pub struct ZipfID;

impl DistanceIDEstimator for ZipfID {
    fn estimate_from_distances<F: Float>(distances: &[F]) -> f64 { zipf_id(distances) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intrinsicdimensionality::KNNIDEstimator;
    use crate::intrinsicdimensionality::test::{
        make_intrinsic_subspace_data, regression_test, test_zeros,
    };

    #[test]
    fn zipf_estimator_regression() {
        regression_test::<ZipfID>(5, 1000, 0, 4.702443328729227);
        regression_test::<ZipfID>(7, 10000, 0, 6.943453727205677);
    }

    #[test]
    fn zipf_estimator_zeros() { test_zeros::<ZipfID>(); }

    #[test]
    fn zipf_estimator_hypersphere_close_to_5() {
        let data = make_intrinsic_subspace_data(1000, 0);
        let table = crate::TableWithDistance::with_distance(&data, crate::distance::Euclidean);
        let tree =
            crate::search::kdtree::KdTree::new(&table, crate::search::kdtree::AxisCycleSplit);

        let estimate = ZipfID::estimate_from_knn(&tree, &table, 0, 100);
        let expected = 4.873777675926932;
        assert!(
            (estimate - expected).abs() < 1e-6,
            "zipf estimate {} deviates from data-based expected {}",
            estimate,
            expected
        );
    }
}
