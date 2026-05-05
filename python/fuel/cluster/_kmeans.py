from .. import _fuel as _fuel
from .._dispatch import _call, _f32_sparse

def _make_kmeans_params(k, max_iter=300, tol=1e-4, seed=None, init=None):
    return _fuel.KMeansParams(
        k,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        init=init,
    )

_KMEANS_VARIANTS = {
    'lloyd':               (_fuel.lloyd_f32,               _fuel.lloyd_f64),
    'lloyd_blas':          (_fuel.lloyd_blas_f32,          _fuel.lloyd_blas_f64),
    'lloyd_naive':         (_fuel.lloyd_naive_f32,         _fuel.lloyd_naive_f64),
    'elkan':               (_fuel.elkan_f32,               _fuel.elkan_f64),
    's_elkan':             (_fuel.simp_elkan_f32,          _fuel.simp_elkan_f64),
    'simp_elkan':          (_fuel.simp_elkan_f32,          _fuel.simp_elkan_f64),
    'simplified_elkan':    (_fuel.simp_elkan_f32,          _fuel.simp_elkan_f64),
    'hamerly':             (_fuel.hamerly_f32,             _fuel.hamerly_f64),
    's_hamerly':           (_fuel.simp_hamerly_f32,        _fuel.simp_hamerly_f64),
    'simp_hamerly':        (_fuel.simp_hamerly_f32,        _fuel.simp_hamerly_f64),
    'simplified_hamerly':  (_fuel.simp_hamerly_f32,        _fuel.simp_hamerly_f64),
    'exponion':            (_fuel.exponion_f32,            _fuel.exponion_f64),
    'shallot':             (_fuel.shallot_f32,             _fuel.shallot_f64),
    'hartigan_wong':       (_fuel.hartigan_wong_f32,       _fuel.hartigan_wong_f64),
    'hartigan_wong_quick': (_fuel.hartigan_wong_quick_f32, _fuel.hartigan_wong_quick_f64),
    'macqueen':            (_fuel.macqueen_f32,            _fuel.macqueen_f64),
    'kmedians':            (_fuel.kmedians_f32,            _fuel.kmedians_f64),
}


def kmeans(data, *, k, variant='simp_elkan', max_iter=300, tol=0,
           seed=None, init=None):
    """K-means clustering (Euclidean space).

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    k : int
        Number of clusters.
    variant : {'lloyd', 'lloyd_blas', 'lloyd_naive', 'elkan', 'simp_elkan',
               'hamerly', 'simp_hamerly', 'exponion', 'shallot',
               'hartigan_wong', 'hartigan_wong_quick', 'macqueen',
               'kmedians'}, default 'simp_hamerly'
        Algorithm variant.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 0
        Convergence tolerance. A value of 0 disables early termination and
        requires actual convergence.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method. Supported values are:
        - 'random'
        - 'first'
        - 'kmeans++'
        - 'kgeometric++'
        - a 2-D NumPy array of shape (k, d) for fixed initial centers.

    Returns
    -------
    tuple
        (centers, assignments, iterations, inertia, inertia_bound)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    v = variant.lower()
    if v not in _KMEANS_VARIANTS:
        raise ValueError(
            f"unknown kmeans variant '{variant}'. Valid: {sorted(_KMEANS_VARIANTS)}"
        )
    f32_fn, f64_fn = _KMEANS_VARIANTS[v]
    return _call(f32_fn, f64_fn, data, params)


def kmedians(data, *, k, max_iter=300, tol=0, seed=None, init=None):
    """K-medians clustering (Euclidean space).

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    k : int
        Number of clusters.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 0
        Convergence tolerance. A value of 0 disables early termination and
        requires actual convergence.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method. Supported values are:
        - 'random'
        - 'first'
        - 'kmeans++'
        - 'kgeometric++'
        - a 2-D NumPy array of shape (k, d) for fixed initial centers.

    Returns
    -------
    tuple
        (centers, assignments, iterations, inertia, inertia_bound)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    return _call(_fuel.kmedians_f32, _fuel.kmedians_f64, data, params)


_KGEOMETRIC_VARIANTS = {
    'default': (_fuel.kgeometric_f32,    _fuel.kgeometric_f64),
    'sh':      (_fuel.kgeometric_sh_f32, _fuel.kgeometric_sh_f64),
}


def kgeometric(data, *, k, steps, variant='default', max_iter=300,
               tol=1e-4, seed=None, init=None):
    """K-geometric-means clustering.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    k : int
        Number of clusters.
    steps : int
        Number of geometric update steps.
    variant : {'default', 'sh'}, default 'default'
        Algorithm variant; 'sh' uses Hamerly-style acceleration.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method.

    Returns
    -------
    tuple
        (centers, assignments, iterations, inertia, inertia_bound)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    v = variant.lower()
    if v not in _KGEOMETRIC_VARIANTS:
        raise ValueError(
            f"unknown kgeometric variant '{variant}'. Valid: {sorted(_KGEOMETRIC_VARIANTS)}"
        )
    f32_fn, f64_fn = _KGEOMETRIC_VARIANTS[v]
    return _call(f32_fn, f64_fn, data, params, steps)


def kgmedians(data, *, k, gamma, alpha, max_iter=300, tol=1e-4,
              seed=None, init=None):
    """Generalised k-medians clustering.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    k : int
        Number of clusters.
    gamma : float
        Gamma parameter for the generalised k-medians objective.
    alpha : float
        Alpha parameter for the generalised k-medians objective.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method.

    Returns
    -------
    tuple
        (centers, assignments, iterations, inertia, inertia_bound)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    return _call(_fuel.kgmedians_f32, _fuel.kgmedians_f64, data, params, gamma, alpha)


def kharmonic(data, *, k, p, max_iter=300, tol=1e-4,
              seed=None, init=None):
    """K-harmonic means clustering.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    k : int
        Number of clusters.
    p : float
        Harmonic power parameter.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method.

    Returns
    -------
    tuple
        (centers, assignments, iterations, inertia, inertia_bound)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    return _call(_fuel.kharmonic_f32, _fuel.kharmonic_f64, data, params, p)


def tkmeans(data, *, k, alpha, max_iter=300, tol=0,
            seed=None, init=None):
    """Trimmed k-means clustering.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    k : int
        Number of clusters.
    alpha : float
        Trimming proportion parameter.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 0
        Convergence tolerance. A value of 0 disables early termination.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method.

    Returns
    -------
    tuple
        (centers, assignments, iterations, inertia, inertia_bound)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    return _call(_fuel.tkmeans_f32, _fuel.tkmeans_f64, data, params, alpha)


def fuzzycmeans(data, *, k, m, max_iter=300, tol=1e-4,
               seed=None, init=None):
    """Fuzzy c-means clustering (Lloyd update).

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    k : int
        Number of clusters.
    m : float
        Fuzziness exponent.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method.

    Returns
    -------
    tuple
        (centers, membership, assignments, iterations, loss)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    return _call(_fuel.fuzzy_lloyd_f32, _fuel.fuzzy_lloyd_f64, data, params, m)


_SPHERICAL_VARIANTS = {
    'lloyd':             (_fuel.spherical_lloyd_f32,       _fuel.spherical_lloyd_f64),
    'elkan':             (_fuel.spherical_elkan_f32,       _fuel.spherical_elkan_f64),
    'selkan':            (_fuel.spherical_simp_elkan_f32,  _fuel.spherical_simp_elkan_f64),
    'simp_elkan':        (_fuel.spherical_simp_elkan_f32,  _fuel.spherical_simp_elkan_f64),
    'simplified_elkan':  (_fuel.spherical_simp_elkan_f32,  _fuel.spherical_simp_elkan_f64),
    'hamerly':           (_fuel.spherical_hamerly_f32,     _fuel.spherical_hamerly_f64),
    's_hamerly':         (_fuel.spherical_simp_hamerly_f32,_fuel.spherical_simp_hamerly_f64),
    'simp_hamerly':      (_fuel.spherical_simp_hamerly_f32,_fuel.spherical_simp_hamerly_f64),
    'simplified_hamerly':(_fuel.spherical_simp_hamerly_f32,_fuel.spherical_simp_hamerly_f64),
}


_SPHERICAL_VARIANTS_SPARSE = {
    'lloyd':              (_fuel.spherical_lloyd_sparse_f32,        _fuel.spherical_lloyd_sparse_f64),
    'elkan':              (_fuel.spherical_elkan_sparse_f32,        _fuel.spherical_elkan_sparse_f64),
    'hamerly':            (_fuel.spherical_hamerly_sparse_f32,      _fuel.spherical_hamerly_sparse_f64),
    'selkan':             (_fuel.spherical_simp_elkan_sparse_f32,   _fuel.spherical_simp_elkan_sparse_f64),
    'simp_elkan':         (_fuel.spherical_simp_elkan_sparse_f32,   _fuel.spherical_simp_elkan_sparse_f64),
    'simplified_elkan':   (_fuel.spherical_simp_elkan_sparse_f32,   _fuel.spherical_simp_elkan_sparse_f64),
    'shamerly':           (_fuel.spherical_simp_hamerly_sparse_f32, _fuel.spherical_simp_hamerly_sparse_f64),
    'simp_hamerly':       (_fuel.spherical_simp_hamerly_sparse_f32, _fuel.spherical_simp_hamerly_sparse_f64),
    'simplified_hamerly': (_fuel.spherical_simp_hamerly_sparse_f32, _fuel.spherical_simp_hamerly_sparse_f64),
}


def spherical_kmeans(data, *, k, variant='simp_elkan', max_iter=300,
                      tol=0, seed=None, init=None):
    """Spherical k-means clustering (cosine distance).

    Accepts either a dense NumPy array or a CSR sparse matrix. The wrapper
    dispatches to the sparse implementation when the provided data has a
    sparse `.data` dtype.

    Parameters
    ----------
    data : ndarray (n, d) or CSR sparse matrix
        Input data matrix.
    k : int
        Number of clusters.
    variant : {'lloyd', 'elkan', 'simp_elkan', 'hamerly', 'simp_hamerly'},
        default 'simp_elkan'
        Algorithm variant.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 0
        Convergence tolerance. A value of 0 disables early termination and
        requires actual convergence.
    seed : int or None, default None
        Optional RNG seed for reproducibility.
    init : str, ndarray, or None, default None
        Initialization method.

    Returns
    -------
    tuple
        (centers, assignments, iterations, inertia, inertia_bound)
    """
    params = _make_kmeans_params(k, max_iter=max_iter, tol=tol,
                                 seed=seed, init=init)
    v = variant.lower()
    if v not in _SPHERICAL_VARIANTS:
        raise ValueError(
            f"unknown spherical_kmeans variant '{variant}'. Valid: {sorted(_SPHERICAL_VARIANTS)}"
        )
    if _f32_sparse(data):
        f32_fn, f64_fn = _SPHERICAL_VARIANTS_SPARSE[v]
        return f32_fn(data, params)

    f32_fn, f64_fn = _SPHERICAL_VARIANTS[v]
    return _call(f32_fn, f64_fn, data, params)
