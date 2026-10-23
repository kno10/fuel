from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float
from ..search import SearchIndex, _normalize_distance_name, _prepare_search_index

# Brute-force variants (no tree required).
_BRUTE_FORCE_VARIANTS = frozenset({'hdbscan_prim', 'slink_hdbscan'})

# Tree-accelerated variants without slack.
_TREE_VARIANTS = frozenset({
    'heap_of_searchers_hdbscan',
    'restarting_search_hdbscan',
    'boruvka_searchers_hdbscan',
})

# Tree-accelerated variants that also require slack.
_TREE_SLACK_VARIANTS = frozenset({
    'buffered_search_hdbscan',
    'lazy_buffered_search_hdbscan',
})

_ALL_VARIANTS = sorted(_BRUTE_FORCE_VARIANTS | _TREE_VARIANTS | _TREE_SLACK_VARIANTS)

_BRUTE_FORCE_DISPATCH = {
    'hdbscan_prim':  (_fuel.hdbscan_prim_f32,  _fuel.hdbscan_prim_f64),
    'slink_hdbscan': (_fuel.slink_hdbscan_f32, _fuel.slink_hdbscan_f64),
}

_TREE_DISPATCH = {
    'heap_of_searchers_hdbscan':   (_fuel.heap_of_searchers_hdbscan_f32,   _fuel.heap_of_searchers_hdbscan_f64),
    'restarting_search_hdbscan':   (_fuel.restarting_search_hdbscan_f32,   _fuel.restarting_search_hdbscan_f64),
    'boruvka_searchers_hdbscan':   (_fuel.boruvka_searchers_hdbscan_f32,   _fuel.boruvka_searchers_hdbscan_f64),
}

_TREE_SLACK_DISPATCH = {
    'buffered_search_hdbscan':      (_fuel.buffered_search_hdbscan_f32,      _fuel.buffered_search_hdbscan_f64),
    'lazy_buffered_search_hdbscan': (_fuel.lazy_buffered_search_hdbscan_f32, _fuel.lazy_buffered_search_hdbscan_f64),
}

def _coresg_index(data, index, distance):
    if not isinstance(index, SearchIndex):
        raise TypeError("CORE-SG requires a reusable SearchIndex instance")
    return _prepare_search_index(data, index, distance=distance, priority_search=True)


class CoreSG:
    """A reusable CORE-SG structure for several HDBSCAN ``min_points`` values."""

    def __init__(self, inner, distance):
        self._inner = inner
        self._distance = distance

    @property
    def max_points(self):
        """Largest supported ``min_points`` value used to build the graph."""
        return self._inner.max_points

    @property
    def num_edges(self):
        """Number of undirected edges in the sparse graph."""
        return self._inner.num_edges

    @property
    def num_points(self):
        """Number of points represented in the graph."""
        return self._inner.num_points

    def __len__(self):
        return self._inner.num_edges

    def neighbors(self, source):
        """Iterate over ``(target, distance)`` entries from ``source``."""
        return iter(self._inner.neighbors(source))

    def core_distance(self, point):
        """Return the stored maximum-``min_points`` core distance for a point."""
        return self._inner.core_distance(point)

    def hdbscan(self, data, min_points, *, index):
        """Build an HDBSCAN hierarchy for ``min_points`` using ``index``."""
        index = _coresg_index(data, index, self._distance)
        return self._inner.hdbscan(data, min_points, index=index)


def coresg(data, max_points, *, index, distance='euclidean'):
    """Build a reusable CORE-SG graph for HDBSCAN.

    Build once at the largest desired ``min_points``, then call
    :meth:`CoreSG.hdbscan` for any positive value up to ``max_points``.
    The returned hierarchy supports the same extraction methods as
    :func:`hdbscan` results.

    Parameters
    ----------
    data : ndarray (n, d)
        Dense feature matrix.
    max_points : int
        Largest ``min_points`` setting to support. Includes the query point
        itself, as in :func:`hdbscan`.
    distance : str, default='euclidean'
        Distance function used for the data.
    index : SearchIndex
        kNN index for ``data``. This same index must be passed to ``hdbscan``
        when deriving a hierarchy.

    Returns
    -------
    CoreSG
        Reusable sparse graph with edge inspection and ``hdbscan(data,
        min_points, index=index)``.
    """
    distance = _normalize_distance_name(distance)
    index = _coresg_index(data, index, distance)
    inner = _call(_fuel.coresg_graph_f32, _fuel.coresg_graph_f64,
                  data, max_points, distance=distance, index=index)
    return CoreSG(inner, _normalize_distance_name(distance))


def hdbscan(data, min_points, variant='hdbscan_prim', *, distance='euclidean',
            slack=5, index=None):
    """
    HDBSCAN hierarchy construction.

    Parameters
    ----------
    data : ndarray (n, d)
    min_points : int
        Minimum number of points for core-distance computation (minPts).
    variant : str
        Algorithm variant. One of:

        Brute-force (O(n^2), any distance):
        - 'hdbscan_prim'
            Prim's MST on mutual reachability distances.
        - 'slink_hdbscan'
            SLINK-style linear-memory variant.

        Tree-accelerated:
        - 'heap_of_searchers_hdbscan'
        - 'restarting_search_hdbscan'
        - 'boruvka_searchers_hdbscan'

        Tree-accelerated with slack buffer:
        - 'buffered_search_hdbscan'
        - 'lazy_buffered_search_hdbscan'

    distance : str
        Distance function name. Default ``'euclidean'``. Accepted names: euclidean,
        sqeuclidean, manhattan, chebyshev, cosine, arccosine, canberra,
        braycurtis, hellinger, clark, chi, chi_squared, jensen_shannon,
        jeffrey, histogram_intersection.
    slack : int
        Buffer slack for buffered_search_hdbscan and
        lazy_buffered_search_hdbscan variants.

    Returns
    -------
    HdbscanHierarchy object with methods:
        core_distances() -> ndarray
        to_scipy_linkage() -> ndarray (n-1, 4)
        extract_clusters_with_noise(num_clusters, min_cluster_size) -> ndarray
        extract_simplified(min_cluster_size) -> dict
        extract_hdbscan(min_cluster_size, hierarchical) -> dict
    """
    v = variant.lower()

    if v in _BRUTE_FORCE_DISPATCH:
        if index is not None:
            raise ValueError(f"variant '{v}' does not use index")
        f32_fn, f64_fn = _BRUTE_FORCE_DISPATCH[v]
        return _call(f32_fn, f64_fn, data, min_points, distance)

    if v in _TREE_DISPATCH:
        index = _prepare_search_index(data, index, distance=distance, priority_search=True)
        f32_fn, f64_fn = _TREE_DISPATCH[v]
        return _call(f32_fn, f64_fn, data, min_points,
                     distance=distance, index=index)

    if v in _TREE_SLACK_DISPATCH:
        index = _prepare_search_index(data, index, distance=distance, priority_search=True)
        f32_fn, f64_fn = _TREE_SLACK_DISPATCH[v]
        return _call(f32_fn, f64_fn, data, min_points, slack,
                     distance=distance, index=index)

    raise ValueError(
        f"unknown HDBSCAN variant '{variant}'. Valid: {_ALL_VARIANTS}"
    )
