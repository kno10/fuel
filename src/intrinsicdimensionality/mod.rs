use crate::api::IndexQuery;

/// Intrinsic dimensionality estimators.
///
/// This module exposes multiple algorithms for local ID estimation from neighborhood distances:
///
/// - `aggregated_hill_id`: aggregated Hill maximum-likelihood tail index.
/// - `hill_id`: classic Hill estimator.
/// - `method_of_moments_id`: moment ratio estimator.
/// - `lmoments_id`: L-moments-based estimator.
/// - `probability_weighted_moments_id`: PWM based estimator.
/// - `probability_weighted_moments_2_id`: extended PWM variant.
/// - `regularly_varying_id`: regular variation quantile estimator.
/// - `zipf_id`: Zipf-weighted log-distance fit.
/// - `generalized_expansion_dimension`: robust median-of-ratios estimator.
/// - `tightlid`: TightLID local geometric estimator.
/// - `abid` / `rabid`: angle-based intrinsic dimensionality estimators.
/// - `alid`: adaptive local intrinsic dimensionality.
///
/// Each estimator provides both a functional API and a wrapper struct implementing
/// the `DistanceIDEstimator` / `KNNIDEstimator` traits (via `*_ID`/`*` names).
pub mod aggregated_hill_estimator;
pub mod alid_estimator;
pub mod angle_based_id;
pub mod generalized_expansion_dimension;
pub mod hill_estimator;
pub mod lmoments_estimator;
pub mod local_pca;
pub mod method_of_moments;
pub mod probability_weighted_moments;
pub mod probability_weighted_moments_2;
pub mod regularly_varying_id;
#[cfg(test)]
mod test;
pub mod tightlid_estimator;
pub mod zipf_estimator;

pub use aggregated_hill_estimator::{AggregatedHillID, aggregated_hill_id};
pub use alid_estimator::{ALID, alid};
pub use angle_based_id::{ABID, RABID, abid, rabid};
pub use generalized_expansion_dimension::{
    GeneralizedExpansionDimension, generalized_expansion_dimension,
};
pub use hill_estimator::{HillID, hill_id};
pub use lmoments_estimator::{LMomentsEstimator, lmoments_id};
pub use local_pca::{LocalPCAID, local_pca_id};
pub use method_of_moments::{MethodOfMoments, method_of_moments_id};
pub use probability_weighted_moments::{
    ProbabilityWeightedMoments, probability_weighted_moments_id,
};
pub use probability_weighted_moments_2::{
    ProbabilityWeightedMoments2, probability_weighted_moments_2_id,
};
pub use regularly_varying_id::{RVEstimator, regularly_varying_id};
pub use tightlid_estimator::{TightLID, tightlid};
pub use zipf_estimator::{ZipfID, zipf_id};

/// kNN-based intrinsic dimensionality estimator API (may require neighbor graph).
pub trait KNNIDEstimator {
    /// Estimate from a kNN searcher around a query point index.
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync;
}

/// Distance-list intrinsic dimensionality estimator API.
pub trait DistanceIDEstimator {
    /// Estimate from a sorted set of nearest-neighbor distances (excluding the query point).
    fn estimate_from_distances<F: crate::Float>(distances: &[F]) -> f64;
}

impl<T> KNNIDEstimator for T
where
    T: DistanceIDEstimator,
{
    /// Every distance-based ID estimator can run for k-nearest neighbors.
    /// Note: we _do_ include the query point itself.
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync,
    {
        let query = data.query().with_index(query_idx);
        let neighbors =
            tree.search_knn(&query, k).into_iter().take(k).map(|n| n.distance).collect::<Vec<_>>();
        Self::estimate_from_distances(&neighbors)
    }
}

/// Find first finite positive distance or len.
pub(crate) fn find_begin<F: crate::Float>(distances: &[F]) -> usize {
    if cfg!(debug_assertions) {
        for window in distances.windows(2) {
            let (a, b) = (window[0], window[1]);
            debug_assert!(a.is_finite(), "distance contains non-finite value");
            debug_assert!(b.is_finite(), "distance contains non-finite value");
            debug_assert!(a <= b, "distance array must be sorted ascending");
        }
    }

    distances.iter().position(|&d| d.is_finite() && d > F::zero()).unwrap_or(distances.len())
}

/// Convert a value to positive finite f64, or return `f64::NAN`.
pub(crate) fn positive_f64<F: crate::Float>(value: F) -> f64 {
    if !value.is_finite() || value <= F::zero() {
        return f64::NAN;
    }
    let value64 = value.to_f64().unwrap_or(f64::NAN);
    if !value64.is_finite() { f64::NAN } else { value64 }
}
