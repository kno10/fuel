import warnings
import numbers

import numpy as np

from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float, _f32
from ..search import pairwise_distances


class KMedoidsResult:
    """K-medoids clustering result.

    Parameters
    ----------
    loss : float
        Loss of this clustering (sum of deviations).
    labels : ndarray
        Cluster assignments.
    medoids : ndarray
        Chosen medoid indices.
    n_iter : int, optional
        Number of iterations.
    n_swap : int, optional
        Number of swaps performed.
    """

    def __init__(self, loss, labels, medoids, n_iter=None, n_swap=None):
        self.loss = loss
        self.labels = labels
        self.medoids = medoids
        self.n_iter = n_iter
        self.n_swap = n_swap

    def __repr__(self):
        return (
            f"KMedoidsResult(loss={self.loss}, labels={self.labels}, "
            f"medoids={self.medoids}, n_iter={self.n_iter}, n_swap={self.n_swap})"
        )


class DynkResult:
    """K-medoids or Silhouette clustering result with automatic number of clusters.

    Parameters
    ----------
    loss : float
        Loss of this clustering (sum of deviations).
    labels : ndarray
        Cluster assignment.
    medoids : ndarray
        Chosen medoid indices.
    bestk : int
        Best k by Medoid Silhouette.
    losses : ndarray
        Medoid Silhouette over range of k.
    rangek : range
        Range of k values tested.
    n_iter : int, optional
        Number of iterations.
    n_swap : int, optional
        Number of swaps performed.
    """

    def __init__(self, loss, labels, medoids, bestk, losses, rangek, n_iter=None, n_swap=None):
        self.loss = loss
        self.labels = labels
        self.medoids = medoids
        self.n_iter = n_iter
        self.n_swap = n_swap
        self.bestk = bestk
        self.losses = losses
        self.rangek = rangek

    def __repr__(self):
        return (
            f"DynkResult(loss={self.loss}, labels={self.labels}, "
            f"medoids={self.medoids}, bestk={self.bestk}, losses={self.losses}, "
            f"rangek={self.rangek}, n_iter={self.n_iter}, n_swap={self.n_swap})"
        )


def _build_kmedoids_result(result):
    if isinstance(result, KMedoidsResult):
        return result
    if not isinstance(result, tuple):
        return result
    if len(result) == 5:
        loss, labels, medoids, n_iter, n_swap = result
        return KMedoidsResult(loss, labels, medoids, n_iter=n_iter, n_swap=n_swap)
    if len(result) == 4:
        loss, labels, medoids, n_iter = result
        return KMedoidsResult(loss, labels, medoids, n_iter=n_iter)
    return result


def _build_dynk_result(result):
    if isinstance(result, DynkResult):
        return result
    if not isinstance(result, tuple):
        return result
    if len(result) == 8:
        loss, labels, medoids, bestk, losses, rangek, n_iter, n_swap = result
        return DynkResult(
            loss, labels, medoids, bestk, losses, rangek,
            n_iter=n_iter, n_swap=n_swap,
        )
    if len(result) == 6:
        loss, labels, medoids, bestk, losses, rangek = result
        return DynkResult(loss, labels, medoids, bestk, losses, rangek)
    return result

_KMEDOIDS_VARIANTS = {
    'fasterpam': (_fuel._fasterpam_f32, _fuel._fasterpam_f64),
    'fastpam1': (_fuel._fastpam1_f32, _fuel._fastpam1_f64),
    'pam': (_fuel._pam_swap_f32, _fuel._pam_swap_f64),
    'alternating': (_fuel._alternating_f32, _fuel._alternating_f64),
}

_KMEDOIDS_MSC_VARIANTS = {
    'pamsil': (_fuel._pamsil_swap_f32, _fuel._pamsil_swap_f64),
    'pammedsil': (_fuel._pammedsil_swap_f32, _fuel._pammedsil_swap_f64),
    'fastmsc': (_fuel._fastmsc_f32, _fuel._fastmsc_f64),
    'fastermsc': (_fuel._fastermsc_f32, _fuel._fastermsc_f64),
}


def _check_medoids(data, medoids, init, seed, distance):
    if isinstance(medoids, np.ndarray):
        if seed is not None:
            warnings.warn("seed will be ignored when initial medoids are given as an array")
        if medoids.ndim != 1:
            raise ValueError("medoids array must be one-dimensional")
        medoids = medoids.astype(np.uintp, copy=False)
        n = pairwise_distances(data, distance, form='square').shape[0]
        if medoids.size > 0 and medoids.max() >= n:
            raise ValueError("Initial medoid indices must be within the number of observations")
        return medoids

    if isinstance(medoids, numbers.Integral):
        k = int(medoids)
        if k <= 0:
            raise ValueError("medoids must be a positive integer")

        square = pairwise_distances(data, distance, form='square')
        if k > square.shape[0]:
            raise ValueError("Number of medoids cannot exceed number of observations")

        if init.lower() == 'build':
            if _f32(square):
                _, _, meds, _ = _fuel._pam_build_f32(square, k)
            else:
                _, _, meds, _ = _fuel._pam_build_f64(square, k)
            return meds.astype(np.uintp, copy=False)

        if init.lower() == 'first':
            return np.arange(k, dtype=np.uintp)

        if init.lower() != 'random':
            raise ValueError("init must be one of 'build', 'first', or 'random'")

        rng = np.random.mtrand._rand if seed is None else np.random.RandomState(seed)
        return rng.choice(square.shape[0], k, False).astype(np.uintp)

    raise ValueError("Specify the number of medoids, or give a numpy array of initial medoids")


def kmedoids(data, medoids, *, variant='par_fasterpam', max_iter=300, seed=None,
             n_cpu=-1, init='random', distance='euclidean'):
    """K-medoids clustering on a distance matrix or feature dataset.

    Parameters
    ----------
    data : ndarray
        Input data array of shape (n, d), a square distance matrix, or a
        condensed distance vector when ``distance='precomputed'``.
    medoids : int or ndarray
        Number of medoids to select or initial medoid indices.
    variant : str, default 'par_fasterpam'
        Algorithm variant. Supported values:
        'pam_build', 'fasterpam', 'rand_fasterpam', 'par_fasterpam',
        'fastpam1', 'pam', 'alternating'.
    max_iter : int, default 300
        Maximum number of iterations.
    seed : int or None, default None
        Random seed. Used for ``init='random'`` and for randomized algorithm
        variants ('rand_fasterpam', 'par_fasterpam').
    n_cpu : int, default -1 (automatic)
        Number of threads for 'par_fasterpam'.
    init : {'build', 'first', 'random'}, default 'random'
        How to initialize medoids when ``medoids`` is an integer.
    distance : str, default 'euclidean'
        Distance metric to compute a pairwise distance matrix from data.
        If ``'precomputed'``, ``data`` is assumed to already be a distance matrix.

    Returns
    -------
    KMedoidsResult
        Named result object with `loss`, `labels`, `medoids`, `n_iter`, and
        `n_swap`.
    """
    data = _ensure_float(data)
    medoids = _check_medoids(data, medoids, init, seed, distance)
    dist = pairwise_distances(data, distance, form='condensed')
    variant = variant.lower()
    if variant == 'pam_build':
        if not isinstance(medoids, numbers.Integral):
            raise ValueError("pam_build variant requires an integer number of medoids")
        k = int(medoids)
        if k <= 0:
            raise ValueError("medoids must be a positive integer")
        square = pairwise_distances(data, distance, form='square')
        if _f32(square):
            return _build_kmedoids_result(_fuel._pam_build_f32(square, k))
        return _build_kmedoids_result(_fuel._pam_build_f64(square, k))

    if variant in _KMEDOIDS_VARIANTS:
        f32_fn, f64_fn = _KMEDOIDS_VARIANTS[variant]
        return _build_kmedoids_result(_call(f32_fn, f64_fn, dist, medoids, max_iter))

    rust_seed = seed if seed is not None else 0
    if variant == 'rand_fasterpam':
        return _build_kmedoids_result(
            _call(_fuel._rand_fasterpam_f32, _fuel._rand_fasterpam_f64,
                  dist, medoids, max_iter, rust_seed)
        )

    if variant == 'par_fasterpam':
        # Rust expects usize; n_cpu <= 0 means "all CPUs" which maps to 0 in rayon
        rust_n_cpu = 0 if n_cpu <= 0 else n_cpu
        return _build_kmedoids_result(
            _call(_fuel._par_fasterpam_f32, _fuel._par_fasterpam_f64,
                  dist, medoids, max_iter, rust_seed, rust_n_cpu)
        )

    raise ValueError(
        f"unknown kmedoids variant '{variant}'. Valid: "
        f"{sorted(list(_KMEDOIDS_VARIANTS) + ['pam_build', 'rand_fasterpam', 'par_fasterpam'])}"
    )


def silhouette_clustering(data, medoids, *, variant='fastermsc', max_iter=300,
                            init='random', seed=None,
                            distance='euclidean'):
    """Clustering with (Medoid-) Silhouette optimization.

    These algorithms optimize the (medoid-) silhouette score directly.

    Parameters
    ----------
    data : ndarray
        Input data array of shape (n, d), a square distance matrix, or a
        condensed distance vector when ``distance='precomputed'``.
    medoids : int or ndarray
        Number of medoids to select or initial medoid indices.
    variant : {'pamsil', 'pammedsil', 'fastmsc', 'fastermsc'}, default 'fastmsc'
        Algorithm variant. 'pamsil' and 'pammedsil' are the baseline silhouette
        optimization variants, analogous to how 'pam' is a baseline version of
        'fasterpam'.
    max_iter : int, default 300
        Maximum number of iterations.
    init : {'build', 'first', 'random'}, default 'build'
        How to initialize medoids when ``medoids`` is an integer.
    seed : int or None, default None
        Random seed used for ``init='random'``.
    distance : str, default 'euclidean'
        Distance metric to compute a pairwise distance matrix from data.
        If ``'precomputed'``, ``data`` is assumed to already be a distance matrix.

    Returns
    -------
    KMedoidsResult
        Named result object with `loss`, `labels`, `medoids`, `n_iter`, and
        `n_swap`.
    """
    data = _ensure_float(data)
    medoids = _check_medoids(data, medoids, init, seed, distance)
    dist = pairwise_distances(data, distance, form='condensed')
    variant = variant.lower()
    if variant not in _KMEDOIDS_MSC_VARIANTS:
        raise ValueError(
            f"unknown msc variant '{variant}'. Valid: {sorted(_KMEDOIDS_MSC_VARIANTS)}"
        )
    f32_fn, f64_fn = _KMEDOIDS_MSC_VARIANTS[variant]
    return _build_kmedoids_result(_call(f32_fn, f64_fn, dist, medoids, max_iter))


def dynmsc(data, medoids, *, minimum_k, max_iter=300,
           init='random', seed=None, distance='euclidean'):
    """Dynamic k-medoids clustering, beginning with the given initial medoids and then
    reducing this until minimum_k is reached. The optimum result (according to the
    medoid silhouette) is returned.

    Parameters
    ----------
    data : ndarray
        Input data array of shape (n, d), a square distance matrix, or a
        condensed distance vector when ``distance='precomputed'``.
    medoids : int or ndarray
        Number of medoids to select or initial medoid indices.
    minimum_k : int
        Minimum number of clusters to consider.
    max_iter : int, default 300
        Maximum number of iterations.
    init : {'build', 'first', 'random'}, default 'random'
        How to initialize medoids when ``medoids`` is an integer.
    seed : int or None, default None
        Random seed used for ``init='random'``.
    distance : str, default 'euclidean'
        Distance metric to compute a pairwise distance matrix from data.
        If ``'precomputed'``, ``data`` is assumed to already be a distance matrix.

    Returns
    -------
    DynkResult
        Named result object with `loss`, `labels`, `medoids`, `bestk`, `losses`,
        `rangek`, `n_iter`, and `n_swap`.
    """
    data = _ensure_float(data)
    medoids = _check_medoids(data, medoids, init, seed, distance)
    if minimum_k > len(medoids):
        raise ValueError("minimum_k must be less than or equal to the number of initial medoids")
    dist = pairwise_distances(data, distance, form='square')
    return _build_dynk_result(
        _call(_fuel._dynmsc_f32, _fuel._dynmsc_f64,
              dist, medoids, minimum_k, max_iter)
    )

