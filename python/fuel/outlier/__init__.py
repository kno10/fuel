from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float, _f32


def angle_based_outlier_detection(data, *, kernel='poly2', distance=None):
    return _call(_fuel.angle_based_outlier_detection_f32,
                 _fuel.angle_based_outlier_detection_f64, data, kernel, distance)


def fast_angle_based_outlier_detection(data, k, *, kernel='poly2', seed=None, distance=None):
    return _call(_fuel.fast_angle_based_outlier_detection_f32,
                 _fuel.fast_angle_based_outlier_detection_f64, data, k, kernel, seed, distance)


def locality_based_abod(data, k, l, *, distance=None):
    return _call(_fuel.locality_based_abod_f32,
                 _fuel.locality_based_abod_f64, data, k, l, distance)


def approximate_local_correlation_integral(data, nmin, alpha, g, *, seed=None,
                                           distance=None):
    return _call(_fuel.approximate_local_correlation_integral_f32,
                 _fuel.approximate_local_correlation_integral_f64,
                 data, nmin, alpha, g, seed, distance)


def local_correlation_integral(data, rmax, nmin, alpha, *, seed=None, distance=None):
    return _call(_fuel.local_correlation_integral_f32,
                 _fuel.local_correlation_integral_f64,
                 data, rmax, nmin, alpha, seed, distance)


def correlation_outlier_probabilities(data, k, expect, dist, *, seed=None,
                                      distance=None):
    return _call(_fuel.correlation_outlier_probabilities_f32,
                 _fuel.correlation_outlier_probabilities_f64,
                 data, k, expect, dist, seed, distance)


def db_outlier_score(data, d, *, seed=None, distance=None):
    return _call(_fuel.db_outlier_score_f32, _fuel.db_outlier_score_f64,
                 data, d, seed, distance)


def db_outlier_detection(data, d, p, *, seed=None, distance=None):
    return _call(_fuel.db_outlier_detection_f32, _fuel.db_outlier_detection_f64,
                 data, d, p, seed, distance)


def distance_from_center(data, *, distance=None):
    return _call(_fuel.distance_from_center_f32, _fuel.distance_from_center_f64,
                 data, distance)


def distance_from_origin(data, *, distance=None):
    return _call(_fuel.distance_from_origin_f32, _fuel.distance_from_origin_f64,
                 data, distance)


def dynamic_window_outlier_factor(data, k, delta, *, seed=None, distance=None):
    return _call(_fuel.dynamic_window_outlier_factor_f32,
                 _fuel.dynamic_window_outlier_factor_f64,
                 data, k, delta, seed, distance)


def flexible_lof(data, krefer, kreach, *, seed=None, distance=None):
    return _call(_fuel.flexible_lof_f32, _fuel.flexible_lof_f64,
                 data, krefer, kreach, seed, distance)


def influence_outlier(data, k, m, *, seed=None, distance=None):
    return _call(_fuel.influence_outlier_f32, _fuel.influence_outlier_f64,
                 data, k, m, seed, distance)


def intrinsic_dimensionality_outlier_score(data, k_c, k_r, *, estimator=None,
                                           seed=None, distance=None):
    return _call(_fuel.intrinsic_dimensionality_outlier_score_f32,
                 _fuel.intrinsic_dimensionality_outlier_score_f64,
                 data, k_c, k_r, estimator, seed, distance)


def isolation_forest(data, num_trees, subsample_size, *, seed=None):
    data = _ensure_float(data)
    if _f32(data):
        return _fuel.isolation_forest_f32(data, num_trees, subsample_size, seed)
    return _fuel.isolation_forest_f64(data, num_trees, subsample_size, seed)


def k_nearest_neighbors_outlier(data, k, *, seed=None, distance=None):
    return _call(_fuel.k_nearest_neighbors_outlier_f32,
                 _fuel.k_nearest_neighbors_outlier_f64,
                 data, k, seed, distance)


def k_nearest_neighbors_distance_deviation(data, k, *, seed=None, distance=None):
    return _call(_fuel.k_nearest_neighbors_distance_deviation_f32,
                 _fuel.k_nearest_neighbors_distance_deviation_f64,
                 data, k, seed, distance)


def k_nearest_neighbors_sos(data, k, *, seed=None, distance=None):
    return _call(_fuel.k_nearest_neighbors_sos_f32, _fuel.k_nearest_neighbors_sos_f64,
                 data, k, seed, distance)


def local_density_factor(data, k, h, c, kernel, *, seed=None, distance=None):
    return _call(_fuel.local_density_factor_f32, _fuel.local_density_factor_f64,
                 data, k, h, c, kernel, seed, distance)


def local_density_outlier_factor(data, k, *, seed=None, distance=None):
    return _call(_fuel.local_density_outlier_factor_f32,
                 _fuel.local_density_outlier_factor_f64,
                 data, k, seed, distance)


def local_intrinsic_dimensionality(data, k, *, estimator=None, seed=None,
                                   distance=None):
    return _call(_fuel.local_intrinsic_dimensionality_f32,
                 _fuel.local_intrinsic_dimensionality_f64,
                 data, k, estimator, seed, distance)


def local_isolation_coefficient(data, k, *, seed=None, distance=None):
    return _call(_fuel.local_isolation_coefficient_f32,
                 _fuel.local_isolation_coefficient_f64,
                 data, k, seed, distance)


def local_outlier_factor(data, k, *, seed=None, distance=None):
    return _call(_fuel.local_outlier_factor_f32, _fuel.local_outlier_factor_f64,
                 data, k, seed, distance)


def local_outlier_probabilities(data, k, m, *, seed=None, distance=None):
    return _call(_fuel.local_outlier_probabilities_f32,
                 _fuel.local_outlier_probabilities_f64,
                 data, k, m, seed, distance)


def outlier_detection_independence_neighbor(data, k, *, seed=None, distance=None):
    return _call(_fuel.outlier_detection_independence_neighbor_f32,
                 _fuel.outlier_detection_independence_neighbor_f64,
                 data, k, seed, distance)


def simple_kernel_density_lof(data, k, h, kernel, *, seed=None, distance=None):
    return _call(_fuel.simple_kernel_density_lof_f32,
                 _fuel.simple_kernel_density_lof_f64,
                 data, k, h, kernel, seed, distance)


def simplified_lof(data, k, *, seed=None, distance=None):
    return _call(_fuel.simplified_lof_f32, _fuel.simplified_lof_f64,
                 data, k, seed, distance)


def stochastic_outlier_selection(data, perplexity, *, seed=None, distance=None):
    return _call(_fuel.stochastic_outlier_selection_f32,
                 _fuel.stochastic_outlier_selection_f64,
                 data, perplexity, seed, distance)


def subspace_outlier_degree(data, k, alpha, *, seed=None, distance=None):
    return _call(_fuel.subspace_outlier_degree_f32, _fuel.subspace_outlier_degree_f64,
                 data, k, alpha, seed, distance)


def weighted_knn(data, k, *, seed=None, distance=None):
    return _call(_fuel.weighted_knn_f32, _fuel.weighted_knn_f64,
                 data, k, seed, distance)


def zero(data):
    """Baseline: assigns zero score to every point."""
    data = _ensure_float(data)
    if _f32(data):
        return _fuel.zero_f32(data)
    return _fuel.zero_f64(data)


def random(data, *, seed=None):
    """Baseline: assigns random scores."""
    data = _ensure_float(data)
    if _f32(data):
        return _fuel.random_f32(data, seed)
    return _fuel.random_f64(data, seed)


def connectivity_outlier_factor(data, k, *, seed=None, distance=None):
    return _call(_fuel.connectivity_outlier_factor_f32,
                 _fuel.connectivity_outlier_factor_f64,
                 data, k, seed, distance)


def variance_of_volume(data, k, *, seed=None, distance=None):
    return _call(_fuel.variance_of_volume_f32, _fuel.variance_of_volume_f64,
                 data, k, seed, distance)


def kdeos(data, kmin, kmax, *, kernel='gaussian', min_bandwidth=0.0, scale=1.0,
          idim=None, seed=None, distance=None):
    return _call(_fuel.kdeos_f32, _fuel.kdeos_f64,
                 data, kmin, kmax, kernel, min_bandwidth, scale, idim, seed, distance)


def intrinsic_stochastic_outlier_selection(data, k, *, estimator=None, seed=None,
                                           distance=None):
    return _call(_fuel.intrinsic_stochastic_outlier_selection_f32,
                 _fuel.intrinsic_stochastic_outlier_selection_f64,
                 data, k, estimator, seed, distance)


def locality_based_abod_kernel(data, k, l, *, kernel='poly2', distance=None):
    return _call(_fuel.locality_based_abod_kernel_f32,
                 _fuel.locality_based_abod_kernel_f64,
                 data, k, l, kernel, distance)
