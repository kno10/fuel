from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float, _f32


def dbscan(data, eps, min_points, *, distance=None, seed=None):
    """DBSCAN density-based clustering.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    eps : float
        Epsilon neighborhood radius.
    min_points : int
        Minimum neighborhood size to form a core point.
    distance : str or None, default None
        Distance function name (default: 'euclidean').
    seed : int or None, default None
        Optional RNG seed for the VP-tree.

    Returns
    -------
    ndarray of int64
        Cluster labels per point (-1 = noise).
    """
    return _call(_fuel.dbscan_f32, _fuel.dbscan_f64,
                 data, eps, min_points, distance, seed)


def parallel_dbscan(data, eps, min_points, *, distance=None, seed=None):
    """Parallel DBSCAN density-based clustering.

    Equivalent to :func:`dbscan` but uses a parallel union-find over core
    points for faster execution on multi-core hardware.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    eps : float
        Epsilon neighborhood radius.
    min_points : int
        Minimum neighborhood size to form a core point.
    distance : str or None, default None
        Distance function name (default: 'euclidean').
    seed : int or None, default None
        Optional RNG seed for the VP-tree.

    Returns
    -------
    ndarray of int64
        Cluster labels per point (-1 = noise).
    """
    return _call(_fuel.parallel_dbscan_f32, _fuel.parallel_dbscan_f64,
                 data, eps, min_points, distance, seed)


def optics(data, eps, min_points, *, distance=None, seed=None):
    """OPTICS ordering and reachability computation.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    eps : float
        Maximum reachability distance (used for initial DBSCAN-style labels
        and to bound the neighborhood search).
    min_points : int
        Minimum neighborhood size to form a core point.
    distance : str or None, default None
        Distance function name (default: 'euclidean').
    seed : int or None, default None
        Optional RNG seed for the VP-tree.

    Returns
    -------
    OpticsResultF32 or OpticsResultF64
        Result object with attributes:
        - ``ordering()``     - processing order (index array)
        - ``reachability()`` - reachability distances per point
        - ``core_distance()``- core distances per point (NaN if not core)
        - ``predecessor()``  - predecessor indices per point (-1 if none)
        - ``labels()``       - DBSCAN-style labels from the initial run
        - ``extract_xi(xi, min_points)`` - Xi-based label extraction
    """
    data = _ensure_float(data)
    if _f32(data):
        return _fuel.optics_f32(data, eps, min_points, distance, seed)
    return _fuel.optics_f64(data, eps, min_points, distance, seed)
