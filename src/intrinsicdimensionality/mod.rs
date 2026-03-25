use crate::api::IndexQuery;

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
pub mod tests;
pub mod tightlid_estimator;
pub mod zipf_estimator;

pub use aggregated_hill_estimator::{AggregatedHillEstimator, aggregated_hill_estimate_from_knn};
pub use alid_estimator::ALIDEstimator;
pub use angle_based_id::{ABIDEstimator, RABIDEstimator, abid_estimate, rabid_estimate};
pub use generalized_expansion_dimension::{
    GeneralizedExpansionDimension, generalized_expansion_dimension,
    generalized_expansion_dimension_from_knn,
};
pub use hill_estimator::{HillEstimator, hill_estimate_from_distances, hill_estimate_from_knn};
pub use lmoments_estimator::{LMomentsEstimator, lmoments_estimate_from_knn};
pub use local_pca::{LocalPCA, local_pca_intrinsic_dimensionality};
pub use method_of_moments::{
    MOMEstimator, MethodOfMoments, method_of_moments, method_of_moments_from_knn,
};
pub use probability_weighted_moments::{
    PWMEstimator, ProbabilityWeightedMoments, probability_weighted_moments,
    probability_weighted_moments_from_knn,
};
pub use probability_weighted_moments_2::{
    PWM2Estimator, ProbabilityWeightedMoments2, probability_weighted_moments_2,
    probability_weighted_moments_2_from_knn,
};
pub use regularly_varying_id::{RVEstimator, rv_estimate_from_distances, rv_estimate_from_knn};
#[cfg(test)]
pub use tests::{
    hypersphere_distances, make_hypersphere_embedded_data, regression_test, test_zeros,
};
pub use tightlid_estimator::{TightLIDEstimator, tightlid_estimate_from_knn};
pub use zipf_estimator::{ZipfEstimator, zipf_estimate_from_knn};

/// kNN-based intrinsic dimensionality estimator API (may require neighbor graph).
pub trait KnnBasedIntrinsicDimensionalityEstimator {
    /// Estimate from a kNN searcher around a query point index.
    fn estimate_from_knn<'a, S, D, F>(tree: &S, data: &'a D, query_idx: usize, k: usize) -> f64
    where
        F: crate::Float,
        D: crate::DistanceData<F> + 'a,
        S: crate::KnnSearch<F, D::Query<'a>> + Sync;
}

/// Distance-list intrinsic dimensionality estimator API.
pub trait DistanceBasedIntrinsicDimensionalityEstimator {
    /// Estimate from a sorted set of nearest-neighbor distances (excluding the query point).
    fn estimate_from_distances(distances: &[f64]) -> f64;
}

impl<T> KnnBasedIntrinsicDimensionalityEstimator for T
where
    T: DistanceBasedIntrinsicDimensionalityEstimator,
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
        let neighbors = tree
            .search_knn(&query, k)
            .into_iter()
            .take(k)
            .map(|n| n.distance.to_f64().unwrap_or(f64::INFINITY))
            .collect::<Vec<_>>();
        Self::estimate_from_distances(&neighbors)
    }
}
