from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float

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


def hdbscan(data, min_points, variant='hdbscan_prim', *, distance=None,
            sample_size=None, slack=None, seed=None):
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

        Tree-accelerated (require sample_size):
        - 'heap_of_searchers_hdbscan'
        - 'restarting_search_hdbscan'
        - 'boruvka_searchers_hdbscan'

        Tree-accelerated with slack buffer (require sample_size and slack):
        - 'buffered_search_hdbscan'
        - 'lazy_buffered_search_hdbscan'

    distance : str or None
        Distance function (default: euclidean). Accepted names: euclidean,
        sqeuclidean, manhattan, chebyshev, cosine, arccosine, canberra,
        braycurtis, hellinger, clark, chi, chi_squared, jensen_shannon,
        jeffrey, histogram_intersection.
    sample_size : int or None
        VP-tree sample size. Required for tree-accelerated variants.
    slack : int or None
        Buffer slack. Required for buffered_search_hdbscan and
        lazy_buffered_search_hdbscan.
    seed : int or None
        RNG seed for VP-tree construction.

    Returns
    -------
    HdbscanHierarchy object with methods:
        core_distances() -> ndarray
        to_scipy_linkage() -> ndarray (n-1, 4)
        extract_clusters_with_noise(num_clusters, min_cluster_size) -> ndarray
        extract_simplified(min_cluster_size) -> dict
        extract_hdbscan(min_cluster_size, hierarchical) -> dict
    """
    data = _ensure_float(data)
    v = variant.lower()

    if v in _BRUTE_FORCE_DISPATCH:
        f32_fn, f64_fn = _BRUTE_FORCE_DISPATCH[v]
        return _call(f32_fn, f64_fn, data, min_points, distance)

    if v in _TREE_DISPATCH:
        if sample_size is None:
            raise ValueError(f"variant '{v}' requires sample_size")
        f32_fn, f64_fn = _TREE_DISPATCH[v]
        return _call(f32_fn, f64_fn, data, min_points, sample_size, seed, distance)

    if v in _TREE_SLACK_DISPATCH:
        if sample_size is None:
            raise ValueError(f"variant '{v}' requires sample_size")
        if slack is None:
            raise ValueError(f"variant '{v}' requires slack")
        f32_fn, f64_fn = _TREE_SLACK_DISPATCH[v]
        return _call(f32_fn, f64_fn, data, min_points, slack, sample_size, seed, distance)

    raise ValueError(
        f"unknown HDBSCAN variant '{variant}'. Valid: {_ALL_VARIANTS}"
    )
