from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float, _f32
from ..search import _prepare_search_index


def dbscan(data, eps, min_points, *, distance="euclidean", variant="dbscan", index=None):
    """DBSCAN density-based clustering.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    eps : float
        Epsilon neighborhood radius.
    min_points : int
        Minimum neighborhood size to form a core point.
    distance : str, default "euclidean"
        Distance function name.
    variant : {'dbscan', 'parallel'}, default 'dbscan'
        If ``'parallel'``, use the parallel DBSCAN implementation.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

    Returns
    -------
    ndarray of int64
        Cluster labels per point (-1 = noise).
    """
    data = _ensure_float(data)
    index = _prepare_search_index(data, index, distance=distance)
    if variant == 'dbscan':
        return _call(_fuel.dbscan_f32, _fuel.dbscan_f64,
                     data, eps, min_points, distance, index=index)
    if variant == 'parallel':
        return _call(_fuel.parallel_dbscan_f32, _fuel.parallel_dbscan_f64,
                     data, eps, min_points, distance, index=index)
    raise ValueError("unsupported variant: {}".format(variant))


def optics(data, max_eps, min_points, *, distance="euclidean", index=None):
    """OPTICS ordering and reachability computation.

    Parameters
    ----------
    data : ndarray (n, d)
        Input data matrix.
    max_eps : float
        Maximum reachability distance (used for initial DBSCAN-style labels
        and to bound the neighborhood search).
    min_points : int
        Minimum neighborhood size to form a core point.
    distance : str, default "euclidean"
        Distance function name.
    index : SearchIndex or str, optional
        Prebuilt search index or index type name.

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
    index = _prepare_search_index(data, index, distance=distance)
    if _f32(data):
        return _fuel.optics_f32(data, max_eps, min_points, distance, index=index)
    return _fuel.optics_f64(data, max_eps, min_points, distance, index=index)

