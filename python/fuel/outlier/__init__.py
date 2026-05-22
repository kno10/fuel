from collections import namedtuple as _namedtuple

from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float, _f32
from ..search import SearchIndex, _prepare_search_index

OutlierResult = _namedtuple('OutlierResult', ['scores', 'metadata'])
"""Result of an outlier detection algorithm.

Attributes
----------
scores : ndarray of shape (n,)
    Per-point outlier scores. Whether higher or lower indicates more outlying
    depends on the algorithm; see ``metadata['ascending']``.
metadata : dict
    Algorithm metadata with the following keys:

    - ``'label'`` : str - Algorithm name.
    - ``'ascending'`` : bool - If True, higher score = more outlying.
    - ``'baseline'`` : float - Expected score for a random point.
    - ``'minimum'`` : float - Minimum score in this result.
    - ``'maximum'`` : float - Maximum score in this result.
    - ``'theoretical_minimum'`` : float - Theoretical minimum score.
    - ``'theoretical_maximum'`` : float - Theoretical maximum score.
"""


def _wrap(result):
    scores, meta = result
    return OutlierResult(scores, meta)


def angle_based_outlier_detection(data, *, kernel='poly2', distance='euclidean'):
    """Angle-Based Outlier Detection (ABOD).

    Parameters
    ----------
    data : array of shape (n, d)
    kernel : {'poly2', 'poly3', 'linear'}
        Kernel function for angle weighting.
    distance : str
        Distance metric name.

    Returns
    -------
    scores : 1-D float array; lower variance = more outlying (ascending=False).
    metadata : dict
    """
    return _wrap(_call(_fuel.angle_based_outlier_detection_f32,
                 _fuel.angle_based_outlier_detection_f64, data, kernel, distance=distance))


def fast_angle_based_outlier_detection(data, k, *, kernel='poly2', distance='euclidean', index=None):
    """Fast Angle-Based Outlier Detection (FastABOD).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    kernel : {'poly2', 'poly3', 'linear'}
        Kernel function for angle weighting.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array; lower variance = more outlying (ascending=False).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.fast_angle_based_outlier_detection_f32,
                 _fuel.fast_angle_based_outlier_detection_f64,
                 data, k, kernel, distance=distance, index=index))


def lb_abod(data, k, l, *, distance='euclidean'):
    """Lower-Bound ABOD (LB-ABOD).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors for candidate set.
    l : int
        Candidate set size for LB approximation.
    distance : str
        Distance metric name.

    Returns
    -------
    scores : 1-D float array (ascending=False).
    metadata : dict
    """
    return _wrap(_call(_fuel.lb_abod_f32,
                 _fuel.lb_abod_f64, data, k, l, distance))


def approximate_local_correlation_integral(data, nmin, alpha, g, *, seed=None,
                                           distance='euclidean'):
    """Approximate Local Correlation Integral (ALOCI).

    Parameters
    ----------
    data : array of shape (n, d)
    nmin : int
        Minimum neighborhood size.
    alpha : float
        Smoothing parameter.
    g : float
        Kernel exponent.
    seed : int or None
        RNG seed for ALOCI sampling.
    distance : str
        Distance metric name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    return _wrap(_call(_fuel.approximate_local_correlation_integral_f32,
                 _fuel.approximate_local_correlation_integral_f64,
                 data, nmin, alpha, g, seed, distance=distance))


def local_correlation_integral(data, rmax, nmin, alpha, *, distance='euclidean', index=None):
    """Local Correlation Integral (LOCI).

    Parameters
    ----------
    data : array of shape (n, d)
    rmax : float
        Radius threshold.
    nmin : int
        Minimum neighborhood size.
    alpha : float
        Smoothing parameter.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.local_correlation_integral_f32,
                 _fuel.local_correlation_integral_f64,
                 data, rmax, nmin, alpha, distance, index=index))


def correlation_outlier_probabilities(data, k, expect, dist, *, distance='euclidean', index=None):
    """Correlation Outlier Probabilities (COP).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    expect : float
        Expected neighbor count.
    dist : {'chi2', 'gamma'}
        Distribution assumption for COP score.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.correlation_outlier_probabilities_f32,
                 _fuel.correlation_outlier_probabilities_f64,
                 data, k, expect, dist, distance, index=index))


def db_outlier_score(data, d, *, distance='euclidean', index=None):
    """DB-Outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    d : float
        Distance threshold.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.db_outlier_score_f32, _fuel.db_outlier_score_f64,
                 data, d, distance, index=index))


def db_outlier_detection(data, d, p, *, distance='euclidean', index=None):
    """DB-Outlier detection.

    Parameters
    ----------
    data : array of shape (n, d)
    d : float
        Distance threshold.
    p : float
        Fraction of neighbors within distance d required to be considered an inlier.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.db_outlier_detection_f32, _fuel.db_outlier_detection_f64,
                 data, d, p, distance, index=index))


def distance_from_center(data, *, distance='euclidean'):
    """Distance from centroid outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    distance : str
        Distance metric name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    return _wrap(_call(_fuel.distance_from_center_f32, _fuel.distance_from_center_f64,
                 data, distance))


def distance_from_origin(data, *, distance='euclidean'):
    """Distance from origin outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    distance : str
        Distance metric name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    return _wrap(_call(_fuel.distance_from_origin_f32, _fuel.distance_from_origin_f64,
                 data, distance))


def dynamic_window_outlier_factor(data, k, delta, *, distance='euclidean', index=None):
    """Dynamic Window Outlier Factor (DWOF).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    delta : float
        Window size parameter.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.dynamic_window_outlier_factor_f32,
                 _fuel.dynamic_window_outlier_factor_f64,
                 data, k, delta, distance, index=index))


def flexible_lof(data, krefer, kreach, *, distance='euclidean', index=None):
    """Flexible Local Outlier Factor.

    Parameters
    ----------
    data : array of shape (n, d)
    krefer : int
        Reference set size.
    kreach : int
        Reachability count.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.flexible_lof_f32, _fuel.flexible_lof_f64,
                 data, krefer, kreach, distance, index=index))


def influence_outlier(data, k, m, *, distance='euclidean', index=None):
    """Influence outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    m : float
        Influence exponent.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.influence_outlier_f32, _fuel.influence_outlier_f64,
                 data, k, m, distance, index=index))


def intrinsic_dimensionality_outlier_score(data, k_c, k_r, *, estimator=None,
                                           distance='euclidean', index=None):
    """Intrinsic Dimensionality Outlier Score (IDOS).

    Parameters
    ----------
    data : array of shape (n, d)
    k_c : int
        Reference neighborhood size.
    k_r : int
        Reachability neighborhood size.
    estimator : str or None
        LID estimator name (see LID estimators table). ``None`` uses the default.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.intrinsic_dimensionality_outlier_score_f32,
                 _fuel.intrinsic_dimensionality_outlier_score_f64,
                 data, k_c, k_r, estimator, distance=distance, index=index))


def isolation_forest(data, num_trees, subsample_size, *, seed=None):
    """Isolation Forest.

    Parameters
    ----------
    data : array of shape (n, d)
    num_trees : int
        Number of isolation trees.
    subsample_size : int
        Subsample size per tree.
    seed : int or None
        RNG seed for tree construction.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    if _f32(data):
        return _wrap(_fuel.isolation_forest_f32(data, num_trees, subsample_size, seed))
    return _wrap(_fuel.isolation_forest_f64(data, num_trees, subsample_size, seed))


def k_nearest_neighbors_outlier(data, k, *, distance='euclidean', index=None):
    """kNN distance outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.k_nearest_neighbors_outlier_f32,
                 _fuel.k_nearest_neighbors_outlier_f64,
                 data, k, distance, index=index))


def k_nearest_neighbors_distance_deviation(data, k, *, distance='euclidean', index=None):
    """kNN Distance Deviation (kNNDD) outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.k_nearest_neighbors_distance_deviation_f32,
                 _fuel.k_nearest_neighbors_distance_deviation_f64,
                 data, k, distance, index=index))


def k_nearest_neighbors_sos(data, k, *, distance='euclidean', index=None):
    """kNN Stochastic Outlier Selection (kNN-SOS).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.k_nearest_neighbors_sos_f32, _fuel.k_nearest_neighbors_sos_f64,
                 data, k, distance, index=index))


def local_density_factor(data, k, h, c, *, kernel, distance='euclidean', index=None):
    """Local Density Factor (LDF).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    h : float
        Bandwidth parameter.
    c : float
        Kernel scaling parameter.
    kernel : str
        Kernel density function name (see kernel names table).
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.local_density_factor_f32, _fuel.local_density_factor_f64,
                 data, k, h, c, kernel, distance, index=index))


def local_density_outlier_factor(data, k, *, distance='euclidean', index=None):
    """Local Density Outlier Factor (LDOF).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.local_density_outlier_factor_f32,
                 _fuel.local_density_outlier_factor_f64,
                 data, k, distance, index=index))


def local_intrinsic_dimensionality(data, k, *, estimator=None, distance='euclidean', index=None):
    """Local Intrinsic Dimensionality (LID) outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    estimator : str or None
        LID estimator name (see LID estimators table). ``None`` uses the default.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.local_intrinsic_dimensionality_f32,
                 _fuel.local_intrinsic_dimensionality_f64,
                 data, k, estimator, distance=distance, index=index))


def local_isolation_coefficient(data, k, *, distance='euclidean', index=None):
    """Local Isolation Coefficient (LIC).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.local_isolation_coefficient_f32,
                 _fuel.local_isolation_coefficient_f64,
                 data, k, distance, index=index))


def local_outlier_factor(data, k, *, distance='euclidean', index=None):
    """Local Outlier Factor (LOF).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name. Pass a precomputed
        ``SearchIndex(data, precompute=k+1)`` for efficient parameter sweeps.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.local_outlier_factor_f32, _fuel.local_outlier_factor_f64,
                 data, k, distance, index=index))


def local_outlier_probabilities(data, k, m, *, distance='euclidean', index=None):
    """Local Outlier Probabilities (LoOP).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    m : float
        Smoothing parameter (lambda).
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array in [0, 1] (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.local_outlier_probabilities_f32,
                 _fuel.local_outlier_probabilities_f64,
                 data, k, m, distance, index=index))


def outlier_detection_independence_neighbor(data, k, *, distance='euclidean', index=None):
    """Outlier Detection using Indegree Number (ODIN).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=False).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.outlier_detection_independence_neighbor_f32,
                 _fuel.outlier_detection_independence_neighbor_f64,
                 data, k, distance, index=index))


def simple_kernel_density_lof(data, k, h, *, kernel, distance='euclidean', index=None):
    """Simple Kernel Density LOF (KDEOS).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    h : float
        Bandwidth.
    kernel : str
        Kernel density function name (see kernel names table).
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.simple_kernel_density_lof_f32,
                 _fuel.simple_kernel_density_lof_f64,
                 data, k, h, kernel, distance, index=index))


def simplified_lof(data, k, *, distance='euclidean', index=None):
    """Simplified Local Outlier Factor.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.simplified_lof_f32, _fuel.simplified_lof_f64,
                 data, k, distance, index=index))


def stochastic_outlier_selection(data, perplexity, *, distance='euclidean', index=None):
    """Stochastic Outlier Selection (SOS).

    Parameters
    ----------
    data : array of shape (n, d)
    perplexity : float
        Effective neighbor count controlling the affinity scale.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.stochastic_outlier_selection_f32,
                 _fuel.stochastic_outlier_selection_f64,
                 data, perplexity, distance, index=index))


def subspace_outlier_degree(data, k, alpha, *, distance='euclidean', index=None):
    """Subspace Outlier Degree (SOD).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    alpha : float
        Subspace balance parameter.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.subspace_outlier_degree_f32, _fuel.subspace_outlier_degree_f64,
                 data, k, alpha, distance, index=index))


def weighted_knn(data, k, *, distance='euclidean', index=None):
    """Weighted kNN outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.weighted_knn_f32, _fuel.weighted_knn_f64,
                 data, k, distance, index=index))


def zero(data):
    """Baseline: assigns zero score to every point."""
    data = _ensure_float(data)
    if _f32(data):
        return _wrap(_fuel.zero_f32(data))
    return _wrap(_fuel.zero_f64(data))


def random(data, *, seed=None):
    """Baseline: assigns random scores.

    Parameters
    ----------
    data : array of shape (n, d)
    seed : int or None
        RNG seed.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    if _f32(data):
        return _wrap(_fuel.random_f32(data, seed))
    return _wrap(_fuel.random_f64(data, seed))


def connectivity_outlier_factor(data, k, *, distance='euclidean', index=None):
    """Connectivity Outlier Factor (COF).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.connectivity_outlier_factor_f32,
                 _fuel.connectivity_outlier_factor_f64,
                 data, k, distance, index=index))


def variance_of_volume(data, k, *, distance='euclidean', index=None):
    """Variance of Volume (VOV) outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.variance_of_volume_f32, _fuel.variance_of_volume_f64,
                 data, k, distance, index=index))


def kdeos(data, kmin, kmax, *, kernel='gaussian', min_bandwidth=0.0, scale=1.0,
          idim=None, distance='euclidean', index=None):
    """Kernel Density Estimation Outlier Score (KDEOS).

    Parameters
    ----------
    data : array of shape (n, d)
    kmin : int
        Minimum number of neighbors for bandwidth selection.
    kmax : int
        Maximum number of neighbors for bandwidth selection.
    kernel : str
        Kernel density function name (see kernel names table).
    min_bandwidth : float
        Minimum bandwidth floor.
    scale : float
        Bandwidth scaling factor.
    idim : int or None
        Intrinsic dimensionality override. ``None`` uses estimated value.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.kdeos_f32, _fuel.kdeos_f64,
                 data, kmin, kmax, kernel, min_bandwidth, scale, idim,
                 distance=distance, index=index))


def intrinsic_stochastic_outlier_selection(data, k, *, estimator=None, distance='euclidean', index=None):
    """Intrinsic Stochastic Outlier Selection (ISOS).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors.
    estimator : str or None
        LID estimator name (see LID estimators table). ``None`` uses the default.
    distance : str
        Distance metric name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    return _wrap(_call(_fuel.intrinsic_stochastic_outlier_selection_f32,
                 _fuel.intrinsic_stochastic_outlier_selection_f64,
                 data, k, estimator, distance=distance, index=index))


def lb_abod_kernel(data, k, l, *, kernel='poly2', distance='euclidean'):
    """Lower-Bound ABOD with configurable kernel (LB-ABOD-kernel).

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of nearest neighbors for candidate set.
    l : int
        Candidate set size for LB approximation.
    kernel : {'poly2', 'poly3', 'linear'}
        Kernel function for angle weighting.
    distance : str
        Distance metric name.

    Returns
    -------
    scores : 1-D float array (ascending=False).
    metadata : dict
    """
    return _wrap(_call(_fuel.lb_abod_kernel_f32,
                 _fuel.lb_abod_kernel_f64,
                 data, k, l, kernel, distance=distance))
