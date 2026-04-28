from .. import _fuel as _fuel
from .._dispatch import _call, _ensure_float, _f32

# Linkage sets - aliases are included so validation errors list them all.
_STANDARD_LINKAGES = frozenset({
    'single', 'complete',
    'average', 'group_average', 'weighted_average',
    'centroid', 'median',
    'ward', 'missq',
    'minimum_sum_squares', 'mnssq',
    'minimum_variance_increase', 'mivar',
    'minimum_variance', 'mnvar',
})

_SET_LINKAGES = (
    (_STANDARD_LINKAGES - frozenset({'weighted_average', 'centroid', 'median'}))
    | frozenset({
        'minimax', 'hausdorff', 'medoid',
        'minimum_sum', 'mnsum',
        'minimum_sum_increase', 'misum',
    })
)

_GEOMETRIC_LINKAGES = frozenset({
    'average', 'group_average',
    'centroid',
    'ward', 'missq',
    'minimum_sum_squares', 'mnssq',
    'minimum_variance_increase', 'mivar',
    'minimum_variance', 'mnvar',
})

_VARIANT_LINKAGES = {
    'agnes':             _STANDARD_LINKAGES,
    'anderberg':         _STANDARD_LINKAGES,
    'muellner':          _STANDARD_LINKAGES,
    'nn_chain':          _STANDARD_LINKAGES,
    'set_agnes':         _SET_LINKAGES,
    'set_anderberg':     _SET_LINKAGES,
    'set_muellner':      _SET_LINKAGES,
    'set_nn_chain':      _SET_LINKAGES,
    'geometric_nn_chain':    _GEOMETRIC_LINKAGES,
    'incremental_nn_chain':  _GEOMETRIC_LINKAGES,
}

_FIXED_SINGLE = frozenset({
    'slink',
    'boruvka_searchers_single_link',
    'heap_of_searchers_single_link',
    'restarting_search_single_link',
    'buffered_search_single_link',
    'lazy_buffered_search_single_link',
})
_FIXED_COMPLETE = frozenset({'clink'})

_STANDARD_DISPATCH = {
    'agnes':         (_fuel.agnes_f32,         _fuel.agnes_f64),
    'anderberg':     (_fuel.anderberg_f32,     _fuel.anderberg_f64),
    'muellner':      (_fuel.muellner_f32,      _fuel.muellner_f64),
    'nn_chain':      (_fuel.nn_chain_f32,      _fuel.nn_chain_f64),
    'set_agnes':     (_fuel.set_agnes_f32,     _fuel.set_agnes_f64),
    'set_anderberg': (_fuel.set_anderberg_f32, _fuel.set_anderberg_f64),
    'set_muellner':  (_fuel.set_muellner_f32,  _fuel.set_muellner_f64),
    'set_nn_chain':  (_fuel.set_nn_chain_f32,  _fuel.set_nn_chain_f64),
}

_ALL_VARIANTS = sorted(
    set(_STANDARD_DISPATCH)
    | {'geometric_nn_chain', 'incremental_nn_chain'}
    | _FIXED_SINGLE
    | _FIXED_COMPLETE
)


def _check_linkage(variant, linkage):
    allowed = _VARIANT_LINKAGES.get(variant)
    if allowed is None:
        return
    if linkage.lower() not in allowed:
        raise ValueError(
            f"linkage '{linkage}' is not valid for variant '{variant}'. "
            f"Valid: {sorted(allowed)}"
        )


def hierarchical(data, variant='agnes', linkage='ward', *, distance=None,
                 sample_size=None, slack=None, seed=None):
    """
    Hierarchical / agglomerative clustering.

    Parameters
    ----------
    data : ndarray (n, d)
    variant : str
        Algorithm variant. One of:
        - 'agnes', 'anderberg', 'muellner', 'nn_chain'
            Standard O(n^3) and O(n^2) algorithms; all linkages.
        - 'set_agnes', 'set_anderberg', 'set_muellner', 'set_nn_chain'
            Set-based variants; extended linkage set (minimax, hausdorff,
            medoid, minimum_sum, minimum_sum_increase).
        - 'geometric_nn_chain'
            Nearest-neighbour chain with Euclidean geometry; geometric
            linkages only; ignores distance parameter.
        - 'incremental_nn_chain'
            Incremental search-based variant; geometric linkages;
            requires sample_size; ignores distance parameter.
        - 'slink'
            Sibson SLINK (fixed: single linkage).
        - 'clink'
            Defays CLINK (fixed: complete linkage).
        - 'boruvka_searchers_single_link', 'heap_of_searchers_single_link',
          'restarting_search_single_link'
            Search-based single-link; require sample_size.
        - 'buffered_search_single_link', 'lazy_buffered_search_single_link'
            Search-based single-link with slack buffer; require sample_size
            and slack.
    linkage : str
        Linkage criterion. Ignored for fixed-linkage variants (slink, clink,
        search-based).
    distance : str or None
        Distance function name (default: euclidean). Not used by geometric
        or incremental variants.
    sample_size : int or None
        Required for search-based and incremental variants.
    slack : int or None
        Required for buffered_search_single_link and
        lazy_buffered_search_single_link.
    seed : int or None
        RNG seed for search-based variants.

    Returns
    -------
    MergeHistory object with cut_by_number_of_clusters(), cut_by_height(),
    and to_scipy_linkage() methods.
    """
    data = _ensure_float(data)
    v = variant.lower()
    _check_linkage(v, linkage)

    if v in _FIXED_SINGLE:
        if v == 'slink':
            return _call(_fuel.slink_f32, _fuel.slink_f64, data, distance)
        if sample_size is None:
            raise ValueError(f"variant '{v}' requires sample_size")
        if v == 'boruvka_searchers_single_link':
            return _call(_fuel.boruvka_searchers_single_link_f32,
                         _fuel.boruvka_searchers_single_link_f64,
                         data, sample_size, seed, distance)
        if v == 'heap_of_searchers_single_link':
            return _call(_fuel.heap_of_searchers_single_link_f32,
                         _fuel.heap_of_searchers_single_link_f64,
                         data, sample_size, seed, distance)
        if v == 'restarting_search_single_link':
            return _call(_fuel.restarting_search_single_link_f32,
                         _fuel.restarting_search_single_link_f64,
                         data, sample_size, seed, distance)
        if slack is None:
            raise ValueError(f"variant '{v}' requires slack")
        if v == 'buffered_search_single_link':
            return _call(_fuel.buffered_search_single_link_f32,
                         _fuel.buffered_search_single_link_f64,
                         data, slack, sample_size, seed, distance)
        # lazy_buffered_search_single_link
        return _call(_fuel.lazy_buffered_search_single_link_f32,
                     _fuel.lazy_buffered_search_single_link_f64,
                     data, slack, sample_size, seed, distance)

    if v in _FIXED_COMPLETE:
        return _call(_fuel.clink_f32, _fuel.clink_f64, data, distance)

    if v == 'geometric_nn_chain':
        return _call(_fuel.geometric_nn_chain_f32, _fuel.geometric_nn_chain_f64,
                     data, linkage)

    if v == 'incremental_nn_chain':
        if sample_size is None:
            raise ValueError("variant 'incremental_nn_chain' requires sample_size")
        return _call(_fuel.incremental_nn_chain_f32, _fuel.incremental_nn_chain_f64,
                     data, linkage, sample_size, seed)

    if v not in _STANDARD_DISPATCH:
        raise ValueError(
            f"unknown hierarchical variant '{variant}'. Valid: {_ALL_VARIANTS}"
        )
    f32_fn, f64_fn = _STANDARD_DISPATCH[v]
    return _call(f32_fn, f64_fn, data, linkage, distance)
