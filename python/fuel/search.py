from . import _fuel as _fuel
from ._dispatch import _call, _ensure_float, _f32
import math as _math
import numpy as _np


_KD_DISTANCE_NAMES = {
    'euclidean', 'l2', 'sqeuclidean', 'squared_euclidean',
    'manhattan', 'l1', 'cityblock',
}

_TREE_NAME_MAP = {
    'auto': 'auto',
    'vp': 'vp',
    'vptree': 'vp',
    'cover': 'cover',
    'covertree': 'cover',
    'ct': 'cover',
    'ball': 'cover',
    'balltree': 'cover',
    'kd': 'kd',
    'kdtree': 'kd',
    'linear': 'linear',
    'brute': 'linear',
    'bruteforce': 'linear',
}


def _choose_search_tree(data, distance):
    ncols = data.shape[1] if data.ndim >= 2 else 1
    if ncols >= 32: return 'linear'
    if distance.lower() in _KD_DISTANCE_NAMES and ncols <= 20:
        return 'kd'
    return 'vp'


def _normalize_search_tree_name(tree):
    if tree is None:
        return 'auto'
    if not isinstance(tree, str):
        raise TypeError("tree must be a string or None")
    key = tree.lower().replace('_', '').replace('-', '')
    if key in _TREE_NAME_MAP:
        return _TREE_NAME_MAP[key]
    raise ValueError(
        f"unknown tree '{tree}', valid values are 'auto', 'vp', 'kd', 'cover', 'linear', 'vptree', 'ct', 'covertree', 'brute', or 'brute_force'"
    )


def _condensed_length_to_n(length):
    if length < 0:
        raise ValueError("condensed distance vector length must be non-negative")
    n = (1 + _math.isqrt(1 + 8 * length)) // 2
    if n * (n - 1) // 2 != length:
        raise ValueError(
            "condensed distance vector length must be n*(n-1)/2 for some integer n"
        )
    return n


def _square_to_condensed(square):
    if square.ndim != 2 or square.shape[0] != square.shape[1]:
        raise ValueError("precomputed distance matrix must be square")
    n = square.shape[0]
    return square[_np.tril_indices(n, -1)].copy()


def _condensed_to_square(condensed):
    if condensed.ndim != 1:
        raise ValueError("precomputed condensed distance vector must be one-dimensional")
    n = _condensed_length_to_n(condensed.shape[0])
    square = _np.zeros((n, n), dtype=condensed.dtype)
    tril = _np.tril_indices(n, -1)
    square[tril] = condensed
    square[(tril[1], tril[0])] = condensed
    return square


def _scipy_to_internal_condensed(data):
    """Reorder a scipy upper-triangular condensed vector to internal lower-triangular.

    scipy pdist stores pair (a, b) with a<b at index a*(2n-a-1)//2 + (b-a-1).
    Our internal format stores pair (i, j) with i>j at index i*(i-1)//2 + j.
    """
    n = _condensed_length_to_n(data.shape[0])
    tril_i, tril_j = _np.tril_indices(n, -1)   # i > j
    # scipy index for the same pair (j, i) where j < i
    scipy_idx = tril_j * (2 * n - tril_j - 1) // 2 + (tril_i - tril_j - 1)
    return data[scipy_idx]


def pairwise_distances(data, distance='euclidean', form='condensed'):
    """Compute pairwise distances for a point set or precomputed distance data.

    Parameters
    ----------
    data : ndarray
        Input array of shape (n, d) for raw points, a square matrix for
        precomputed distances, or a condensed distance vector in scipy
        ``pdist`` format (upper-triangular, pairs (a,b) with a<b) when
        ``distance='precomputed'``.
    distance : str, default 'euclidean'
        Distance metric to use. If ``'precomputed'``, ``data`` is assumed to
        already contain pairwise distances.
    form : {'condensed', 'square'}, default 'condensed'
        Output format for the pairwise distances.

    Returns
    -------
    ndarray
        A condensed distance vector when ``form='condensed'`` or a square
        distance matrix when ``form='square'``.
    """
    data = _ensure_float(data)
    distance = distance.lower()
    form = form.lower()

    if form not in {'condensed', 'square'}:
        raise ValueError("form must be 'condensed' or 'square'")

    if distance == 'precomputed':
        if data.ndim == 1:
            # User-supplied condensed vectors are assumed to be in scipy pdist format
            # (upper-triangular, pairs (a,b) with a<b in row-major order).
            # Convert to internal lower-triangular format before further use.
            internal = _scipy_to_internal_condensed(data)
            if form == 'condensed':
                return internal
            return _condensed_to_square(internal)
        if data.ndim == 2:
            if data.shape[0] != data.shape[1]:
                raise ValueError("precomputed distance matrix must be square")
            if form == 'square':
                return data
            return _square_to_condensed(data)
        raise ValueError(
            "precomputed data must be a square matrix or condensed distance vector"
        )

    if data.ndim != 2:
        raise ValueError("data must be a 2D array of observations when distance is not 'precomputed'")

    if form == 'condensed':
        if _f32(data):
            return _fuel._compute_pairwise_distances_condensed_f32(data, distance)
        return _fuel._compute_pairwise_distances_condensed_f64(data, distance)

    if _f32(data):
        return _fuel._compute_pairwise_distances_f32(data, distance)
    return _fuel._compute_pairwise_distances_f64(data, distance)


def _build_search_index(data, tree, distance, seed=None, precompute=None):
    data = _ensure_float(data)
    tree = _normalize_search_tree_name(tree)
    if tree == 'auto':
        tree = _choose_search_tree(data, distance)

    if tree == 'vp':
        if _f32(data):
            return _fuel.build_vp_tree_f32(data, distance, seed, precompute)
        return _fuel.build_vp_tree_f64(data, distance, seed, precompute)
    if tree == 'cover':
        if _f32(data):
            return _fuel.build_cover_tree_f32(data, distance, seed, precompute)
        return _fuel.build_cover_tree_f64(data, distance, seed, precompute)
    if tree == 'kd':
        if _f32(data):
            return _fuel.build_kd_tree_f32(data, distance, precompute)
        return _fuel.build_kd_tree_f64(data, distance, precompute)
    if tree == 'linear':
        if _f32(data):
            return _fuel.build_linear_scan_f32(data, distance, precompute)
        return _fuel.build_linear_scan_f64(data, distance, precompute)
    raise ValueError(
        f"unknown tree '{tree}', valid values are 'auto', 'vp', 'kd', 'cover', 'linear'"
    )


def _prepare_search_index(data, index, *, distance=None, seed=None, priority_search=False):
    if index is None:
        tree = _choose_search_tree(data, distance or 'euclidean')
        if priority_search and tree == 'kd':
            tree = 'vp'
        return _build_search_index(data, tree, distance or 'euclidean', seed)
    if isinstance(index, SearchIndex):
        if index.dtype != data.dtype or index.shape != data.shape:
            raise ValueError(
                "SearchIndex must be built from the same data shape and dtype as the outlier input"
            )
        return index.index
    if isinstance(index, str):
        return _build_search_index(data, index, distance or 'euclidean', seed)
    raise TypeError("index must be a SearchIndex instance or string name")


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

    Parameters
    ----------
    data : array-like of shape (n, d)
        Input data set. Converted to float32 or float64 as needed.
    distance : str, optional
        Distance function name. Default ``'euclidean'``.
    tree : {'auto', 'vp', 'kd', 'cover', 'linear'}, optional
        Index structure to use. ``'auto'`` chooses ``'kd'`` for low-dimensional
        Euclidean-like distances and otherwise chooses ``'vp'``. Aliases
        ``'ct'`` / ``'covertree'`` map to ``'cover'``, ``'vptree'`` maps to
        ``'vp'``, and ``'brute'`` / ``'brute_force'`` map to ``'linear'``.
    seed : int or None, optional
        Random seed for VP-tree or cover-tree construction.
    precompute : int or None, optional
        If provided, precompute kNN results up to this value for repeated
        queries.
    """

    def __init__(self, data, *, distance='euclidean', tree=None, seed=None, precompute=None):
        data = _ensure_float(data)
        self.data = data
        self.distance = distance
        self.seed = seed
        self.precompute = precompute
        self.dtype = data.dtype
        self.shape = data.shape
        self.tree = _normalize_search_tree_name(tree)
        self._source_shape = data.shape
        self._source_strides = data.strides
        self._source_ptr = data.__array_interface__['data'][0]

        if self.tree is None or self.tree == 'auto':
            self.tree = _choose_search_tree(data, distance)
        if self.tree not in {'vp', 'kd', 'cover', 'linear'}:
            raise ValueError(f"unknown tree '{self.tree}', valid values are 'auto', 'vp', 'kd', 'cover', 'linear', or None")

        self.index = None
        if self.tree == 'vp':
            if _f32(data):
                self.index = _fuel.build_vp_tree_f32(data, distance, seed, self.precompute)
            else:
                self.index = _fuel.build_vp_tree_f64(data, distance, seed, self.precompute)
        elif self.tree == 'cover':
            if _f32(data):
                self.index = _fuel.build_cover_tree_f32(data, distance, seed, self.precompute)
            else:
                self.index = _fuel.build_cover_tree_f64(data, distance, seed, self.precompute)
        elif self.tree == 'kd':
            if _f32(data):
                self.index = _fuel.build_kd_tree_f32(data, distance, self.precompute)
            else:
                self.index = _fuel.build_kd_tree_f64(data, distance, self.precompute)
        elif self.tree == 'linear':
            if _f32(data):
                self.index = _fuel.build_linear_scan_f32(data, distance, self.precompute)
            else:
                self.index = _fuel.build_linear_scan_f64(data, distance, self.precompute)

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


def knn_search(data, query, k, *, exclude_self=None, distance='euclidean', tree='auto', seed=None):
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


def range_search(data, query, radius, *, exclude_self=None, distance='euclidean', tree='auto', seed=None):
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
