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
    scores : 1-D float array
    metadata : dict

    Larger scores indicate more anomalous observations.
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


def loda(data, tau=0.01, *, n_bins=None, n_projections=None, seed=None):
    """LODA (Lightweight On-line Detector of Anomalies).

    Parameters
    ----------
    data : array of shape (n, d)
    tau : float, default=0.01
        Early stopping threshold for the projection score convergence.
    n_bins : int or None
        Fixed number of histogram bins per projection. The default selects
        the number of bins automatically.
    n_projections : int or None
        Fixed number of random projections. The default stops when scores
        converge according to ``tau``.
    seed : int or None
        Random seed for reproducible projections.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    if _f32(data):
        return _wrap(_fuel.loda_f32(data, tau, n_bins=n_bins,
                                    n_projections=n_projections, seed=seed))
    return _wrap(_fuel.loda_f64(data, tau, n_bins=n_bins,
                                n_projections=n_projections, seed=seed))


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


def noise_as_outliers(labels):
    """Mark DBSCAN noise points (label == -1) as outliers.

    Parameters
    ----------
    labels : array-like of int
        Cluster labels as returned by DBSCAN (noise = -1).

    Returns
    -------
    scores : 1-D float array (ascending=True); noise points score 1.0.
    metadata : dict
    """
    import numpy as _np
    return _wrap(_fuel.noise_as_outliers_f64(_np.asarray(labels, dtype=_np.int64)))


def silhouette_outlier(data, labels, *, noise_label=None):
    """Silhouette-based outlier score.

    Parameters
    ----------
    data : array of shape (n, d)
    labels : array-like of int
        Cluster assignment labels.
    noise_label : int or None
        Label value that denotes noise/outlier points (excluded from clusters).

    Returns
    -------
    scores : 1-D float64 array (ascending=True).
    metadata : dict
    """
    import numpy as _np
    data = _np.asarray(data, dtype=_np.float64)
    labels = _np.asarray(labels, dtype=_np.int64)
    return _wrap(_fuel.silhouette_outlier_f64(data, labels, noise_label))


def glosh(hierarchy, min_cluster_size):
    """GLOSH outlier score from a pre-computed HDBSCAN hierarchy.

    Parameters
    ----------
    hierarchy : HdbscanHierarchy
        Hierarchy object returned by an HDBSCAN function.
    min_cluster_size : int
        Minimum cluster size used for GLOSH computation.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    return _wrap(hierarchy.glosh(min_cluster_size))


def glosh_from_data(data, min_points, min_cluster_size, *, distance='euclidean'):
    """GLOSH outlier score computed directly from data (brute-force HDBSCAN).

    Parameters
    ----------
    data : array of shape (n, d)
    min_points : int
        Minimum points for core distance.
    min_cluster_size : int
        Minimum cluster size for GLOSH.
    distance : str
        Distance metric name.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    data = _ensure_float(data)
    return _wrap(_call(_fuel.glosh_from_data_f32, _fuel.glosh_from_data_f64,
                 data, min_points, min_cluster_size, distance))


def dbscan_outlier(data, eps, min_points, *, distance='euclidean', index=None):
    """DBSCAN-based outlier score.

    Runs DBSCAN and scores points by whether they are noise or core/border.

    Parameters
    ----------
    data : array of shape (n, d)
    eps : float
        Neighborhood radius.
    min_points : int
        Minimum points to form a core point.
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
    return _wrap(_call(_fuel.dbscan_outlier_f32, _fuel.dbscan_outlier_f64,
                 data, eps, min_points, distance, index=index))


def kmeans_outlier(data, k_or_km, *, rule='distance', max_iter=300, tol=1e-4, seed=None):
    """k-means clustering outlier score.

    Can be called in two ways:

    1. ``kmeans_outlier(data, k, ...)`` -- runs k-means (lloyd) internally.
    2. ``kmeans_outlier(data, km)`` -- uses a precomputed ``KMeansResult``.

    Parameters
    ----------
    data : array of shape (n, d)
    k_or_km : int or KMeansResult
        Number of clusters, or a precomputed k-means result.
    rule : {'distance', 'distance_singletons', 'variance'}
        Scoring rule.
    max_iter : int
        Maximum k-means iterations (only used when k is an int).
    tol : float
        Convergence tolerance (only used when k is an int).
    seed : int or None
        Random seed (only used when k is an int).

    Returns
    -------
    result : OutlierResult
    """
    if hasattr(k_or_km, 'centers'):
        km = k_or_km
        return _wrap(_call(_fuel.kmeans_outlier_from_km_f32, _fuel.kmeans_outlier_from_km_f64,
                     data, km.centers, km.labels, rule))
    return _wrap(_call(_fuel.kmeans_outlier_f32, _fuel.kmeans_outlier_f64,
                 data, k_or_km, rule, max_iter, tol, seed))


def cblof(data, k_or_km, alpha=0.9, beta=3.0, *, max_iter=300, tol=1e-4, seed=None):
    """Cluster-Based Local Outlier Factor (CBLOF).

    Can be called in two ways:

    1. ``cblof(data, k, ...)`` -- runs k-means (lloyd) internally.
    2. ``cblof(data, km)`` -- uses a precomputed ``KMeansResult``.

    Parameters
    ----------
    data : array of shape (n, d)
    k_or_km : int or KMeansResult
        Number of clusters, or a precomputed k-means result.
    alpha : float
        Large-cluster fraction threshold (0 < alpha < 1).
    beta : float
        Size ratio threshold for small-to-large cluster boundary.
    max_iter : int
        Maximum k-means iterations (only used when k is an int).
    tol : float
        Convergence tolerance (only used when k is an int).
    seed : int or None
        Random seed (only used when k is an int).

    Returns
    -------
    result : OutlierResult
    """
    if hasattr(k_or_km, 'centers'):
        km = k_or_km
        return _wrap(_call(_fuel.cblof_from_km_f32, _fuel.cblof_from_km_f64,
                     data, km.centers, km.labels, alpha, beta))
    return _wrap(_call(_fuel.cblof_f32, _fuel.cblof_f64,
                 data, k_or_km, alpha, beta, max_iter, tol, seed))


def tkmeans_outlier(data_or_km, k=None, alpha=0.05, *, max_iter=300, tol=1e-4, seed=None):
    """Trimmed k-means outlier score.

    Points in the trimmed fraction are scored as outliers (score 1), inliers as 0.

    Can be called in two ways:

    1. ``tkmeans_outlier(data, k, alpha=0.05, ...)`` -- runs tkmeans internally.
    2. ``tkmeans_outlier(km)`` -- uses a precomputed ``KMeansResult`` from
       ``tkmeans()``.  The result must have a ``noise_mask`` (i.e. it must come
       from ``tkmeans``, not plain ``kmeans``).

    Parameters
    ----------
    data_or_km : array of shape (n, d) or KMeansResult
        Input data or a precomputed tkmeans result.
    k : int, optional
        Number of clusters (only used when data is passed).
    alpha : float
        Trimming fraction (fraction of points treated as outliers).
    max_iter : int
        Maximum k-means iterations.
    tol : float
        Convergence tolerance.
    seed : int or None
        Random seed.

    Returns
    -------
    result : OutlierResult
        Binary tkmeans outlier scores (trimmed=1, inlier=0).
    """
    if hasattr(data_or_km, 'noise_mask'):
        km = data_or_km
        if km.noise_mask is None:
            raise ValueError(
                "tkmeans_outlier requires a KMeansResult with noise_mask; use tkmeans()"
            )
        if _f32(km.centers):
            return _wrap(_fuel.tkmeans_outlier_from_km_f32(km.noise_mask))
        return _wrap(_fuel.tkmeans_outlier_from_km_f64(km.noise_mask))
    return _wrap(_call(_fuel.tkmeans_outlier_f32, _fuel.tkmeans_outlier_f64,
                 data_or_km, k, alpha, max_iter, tol, seed))


def tkmeans_star_outlier(data, k_or_km, alpha=0.05, *, max_iter=300, tol=1e-4, seed=None):
    """Trimmed k-means* outlier score.

    Like tkmeans_outlier but uses distance-to-nearest-cluster-center for
    non-trimmed points, and noise-mask-based trimmed assignment for outliers.

    Can be called in two ways:

    1. ``tkmeans_star_outlier(data, k, alpha=0.05, ...)`` -- runs tkmeans internally.
    2. ``tkmeans_star_outlier(data, km)`` -- uses a precomputed ``KMeansResult``
       from ``tkmeans()``.  The result must have a ``noise_mask``.

    Parameters
    ----------
    data : array of shape (n, d)
    k_or_km : int or KMeansResult
        Number of clusters, or a precomputed tkmeans result.
    alpha : float
        Trimming fraction.
    max_iter : int
        Maximum k-means iterations (only used when k is an int).
    tol : float
        Convergence tolerance (only used when k is an int).
    seed : int or None
        Random seed (only used when k is an int).

    Returns
    -------
    result : OutlierResult
    """
    if hasattr(k_or_km, 'noise_mask'):
        km = k_or_km
        if km.noise_mask is None:
            raise ValueError(
                "tkmeans_star_outlier requires a KMeansResult with noise_mask; use tkmeans()"
            )
        return _wrap(_call(_fuel.tkmeans_star_outlier_from_km_f32,
                           _fuel.tkmeans_star_outlier_from_km_f64,
                     data, km.centers, km.labels, km.noise_mask))
    return _wrap(_call(_fuel.tkmeans_star_outlier_f32, _fuel.tkmeans_star_outlier_f64,
                 data, k_or_km, alpha, max_iter, tol, seed))


def em_outlier(data, k, *, model='diagonal', max_iter=200, tol=1e-5,
               prior=0.0, noise_ratio=0.0, seed=None):
    """EM mixture model outlier score.

    Scores points by their log-likelihood under the fitted mixture model.

    Parameters
    ----------
    data : array of shape (n, d)
    k : int
        Number of mixture components.
    model : {'diagonal', 'spherical', 'multivariate'}
        Gaussian model type. 'multivariate' requires BLAS.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Log-likelihood convergence delta.
    prior : float
        MAP Dirichlet prior concentration (0 = MLE, 1 = Laplace smoothing).
    noise_ratio : float
        Expected fraction of data treated as noise/outliers (0 disables).
    seed : int or None
        Random seed.

    Returns
    -------
    scores : 1-D float array (ascending=True).
    metadata : dict
    """
    return _wrap(_call(_fuel.em_outlier_f32, _fuel.em_outlier_f64,
                 data, k, model, max_iter, tol, prior, noise_ratio, seed))
