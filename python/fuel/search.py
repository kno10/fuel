from . import _fuel as _fuel
from ._dispatch import _call, _ensure_float, _f32
import numpy as _np


def knn_search(data, query, k, *, exclude_self=None, distance='euclidean', tree='vp', seed=None):
    """Find k nearest neighbors for every point in *query* within *data*.

    Parameters
    ----------
    data : array-like of shape (n, d)
        Input data set. Converted to float32 or float64 as needed.
    query : array-like of shape (m, d)
        Query points. Converted to float32 or float64 as needed.
    k : int
        Number of neighbors to return per query point.
    exclude_self : bool or None, optional
        If True, self-matches are excluded and k distinct neighbors are
        returned when ``query`` refers to the same underlying array as
        ``data``. If False, query points may be included in the results.
        If None (default), self-exclusion is enabled only when ``query`` is the
        same source array as ``data``.
    distance : str, optional
        Distance function name. Default ``'euclidean'``. Supported names:
        ``euclidean`` / ``l2``, ``sqeuclidean`` / ``squared_euclidean``,
        ``manhattan`` / ``l1`` / ``cityblock``, ``chebyshev`` / ``linf`` /
        ``chessboard``, ``cosine``, ``arccosine`` / ``angular``,
        ``canberra``, ``braycurtis`` / ``bray_curtis``, ``hellinger``,
        ``clark``, ``chi``, ``chi_squared`` / ``chisquared`` / ``chi2``,
        ``jensen_shannon`` / ``jensenshannon`` / ``js``,
        ``jeffrey`` / ``jeffreys``, ``histogram_intersection`` /
        ``intersection``. For ``tree='kd'`` only ``'euclidean'``,
        ``'sqeuclidean'``, and ``'manhattan'`` are supported.
    tree : {'auto', 'vp', 'kd', 'cover', 'linear'}, optional
        Index structure to use. ``'auto'`` chooses ``'kd'`` for low-
        dimensional Euclidean-like distances and otherwise chooses ``'vp'``.
        ``'vp'`` supports all distance functions. ``'kd'`` only supports
        coordinate-based distances but can be faster for low-dimensional
        Euclidean data. ``'cover'`` uses a cover tree and supports all
        distances. ``'linear'`` uses exact brute-force linear scan search.
        distances. ``'vp'`` and ``'cover'`` are exact only for metric
        distances.
    seed : int or None, optional
        Random seed for VP-tree or cover-tree construction (ignored when
        ``tree='kd'``).

    Returns
    -------
    indices : ndarray of shape (m, k), dtype int64
        Indices of the k nearest neighbors for each query point. Entries are
        -1 where fewer than k neighbors exist.
    distances : ndarray of shape (m, k)
        Corresponding distances. Entries are ``inf`` where fewer than k
        neighbors exist.
    """
    data = _ensure_float(data)
    query = _ensure_float(query)
    index = SearchIndex(data, distance=distance, tree=tree, seed=seed)
    return index.knn(query, k, exclude_self=exclude_self)


_KD_DISTANCE_NAMES = {
    'euclidean', 'l2', 'sqeuclidean', 'squared_euclidean',
    'manhattan', 'l1', 'cityblock',
}


def _choose_search_tree(data, distance):
    if (
        distance.lower() in _KD_DISTANCE_NAMES
        and data.ndim <= 4
    ):
        return 'kd'
    return 'vp'


def _same_source(data, query):
    return (
        data.shape == query.shape
        and data.strides == query.strides
        and data.__array_interface__['data'][0] == query.__array_interface__['data'][0]
    )


class SearchIndex:
    """Persistent search index wrapper for repeated queries.

    Use :class:`SearchIndex` directly instead of building a separate helper
    wrapper. For repeated queries, construct the index once and call
    ``knn`` or ``radius_search`` multiple times.
    """

    def __init__(self, data, *, distance='euclidean', tree=None, seed=None):
        data = _ensure_float(data)
        self.data = data
        self.distance = distance
        self.seed = seed
        self.dtype = data.dtype
        self.shape = data.shape
        self.tree = tree
        self._source_shape = data.shape
        self._source_strides = data.strides
        self._source_ptr = data.__array_interface__['data'][0]

        if self.tree is None or self.tree == 'auto':
            self.tree = _choose_search_tree(data, distance)
        if self.tree == 'covertree': self.tree = 'cover'
        if self.tree not in {'vp', 'kd', 'cover', 'linear'}:
            raise ValueError(f"unknown tree '{self.tree}', valid values are 'auto', 'vp', 'kd', 'cover', 'linear', or None")

        self.index = None
        if self.tree == 'vp':
            if _f32(data):
                self.index = _fuel.build_vp_tree_f32(data, distance, seed)
            else:
                self.index = _fuel.build_vp_tree_f64(data, distance, seed)
        elif self.tree == 'cover':
            if _f32(data):
                self.index = _fuel.build_cover_tree_f32(data, distance, seed)
            else:
                self.index = _fuel.build_cover_tree_f64(data, distance, seed)
        elif self.tree == 'kd':
            if _f32(data):
                self.index = _fuel.build_kd_tree_f32(data, distance)
            else:
                self.index = _fuel.build_kd_tree_f64(data, distance)
        elif self.tree == 'linear':
            if _f32(data):
                self.index = _fuel.build_linear_scan_f32(data, distance)
            else:
                self.index = _fuel.build_linear_scan_f64(data, distance)

    def _same_source(self, query):
        return (
            query.shape == self._source_shape
            and query.strides == self._source_strides
            and query.__array_interface__['data'][0] == self._source_ptr
        )

    def knn(self, query, k, *, exclude_self=None):
        query = _ensure_float(query)
        if query.dtype != self.dtype:
            query = query.astype(self.dtype)
        if query.ndim != 2 or query.shape[1] != self.shape[1]:
            raise ValueError(
                f"query shape {query.shape} is not compatible with index data shape {self.shape}"
            )
        if exclude_self is None:
            exclude_self = self._same_source(query)
        elif exclude_self and not self._same_source(query):
            raise ValueError(
                "exclude_self=True is only supported when query refers to the same underlying array as data"
            )
        return self.index.knn(self.data, query, self.distance, k, exclude_self=exclude_self)

    def radius_search(self, query, radius, *, exclude_self=None):
        query = _ensure_float(query)
        if query.dtype != self.dtype:
            query = query.astype(self.dtype)
        if query.ndim != 2 or query.shape[1] != self.shape[1]:
            raise ValueError(
                f"query shape {query.shape} is not compatible with index data shape {self.shape}"
            )
        if exclude_self is None:
            exclude_self = self._same_source(query)
        elif exclude_self and not self._same_source(query):
            raise ValueError(
                "exclude_self=True is only supported when query refers to the same underlying array as data"
            )
        return self.index.radius_search(self.data, query, self.distance, radius, exclude_self=exclude_self)

    def __repr__(self):
        return (
            f"<SearchIndex tree={self.tree!r} dtype={self.dtype!r} shape={self.shape!r}>"
        )


def range_search(data, query, radius, *, exclude_self=None, distance='euclidean', seed=None, tree='vp'):
    """Find all neighbors within *radius* for every point in *query* against *data*.

    Parameters
    ----------
    data : array-like of shape (n, d)
        Input data set. Converted to float32 or float64 as needed.
    query : array-like of shape (m, d)
        Query points. Converted to float32 or float64 as needed.
    radius : float
        Search radius. Points at distance <= radius are returned.
    exclude_self : bool or None, optional
        If True, self-matches are excluded. If False, query points may be
        included in the results. If None (default), self-exclusion is enabled
        only when ``query`` refers to the same underlying array as ``data``.
    distance : str, optional
        Distance function name. Default ``'euclidean'``.
    seed : int or None, optional
        Random seed for VP-tree or cover-tree construction.
    tree : {'auto', 'vp', 'kd', 'cover', 'linear'}, optional
        Index structure to use. ``'auto'`` chooses ``'vp'`` for the
        default radius search path. ``'kd'`` supports coordinate-based
        radius search for suitable distances. ``'vp'`` and ``'cover'``
        are exact only for metric distances. ``'linear'`` uses exact
        brute-force linear scan search.
        are exact only for metric distances.

    Returns
    -------
    tuple of two ndarray of dtype object
        ``(indices, distances)`` where each element is an object array of
        1-D ndarrays for each query point. Each inner array contains the
        neighbors for that query, sorted by distance.
    """
    data = _ensure_float(data)
    query = _ensure_float(query)
    index = SearchIndex(data, distance=distance, tree=tree, seed=seed)
    return index.radius_search(query, radius, exclude_self=exclude_self)
