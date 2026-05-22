from collections import namedtuple as _namedtuple

import numpy as np
from .. import _fuel as _fuel

SilhouetteResult = _namedtuple('SilhouetteResult', ['mean', 'stddev', 'values'])


def _ensure_f64(data):
    """Return data as a C-contiguous float64 array."""
    a = np.asarray(data, dtype=np.float64)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return a


def _ensure_i64(labels):
    """Return labels as a C-contiguous int64 array."""
    a = np.asarray(labels, dtype=np.int64)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return a


# ---------------------------------------------------------------------------
# External measures (comparing two label assignments)
# ---------------------------------------------------------------------------

def pair_counting(labels1, labels2, *, self_pairing=False, break_noise_clusters=False,
                  noise_label1=None, noise_label2=None):
    """Pair-counting statistics: F1, precision, recall, ARI, Jaccard, etc."""
    return _fuel.pair_counting(
        _ensure_i64(labels1), _ensure_i64(labels2),
        self_pairing, break_noise_clusters, noise_label1, noise_label2,
    )


def entropy_measures(labels1, labels2, *, self_pairing=False, break_noise_clusters=False,
                     noise_label1=None, noise_label2=None):
    """Entropy-based measures: MI, NMI variants, VI, conditional entropy, etc."""
    return _fuel.entropy_measures(
        _ensure_i64(labels1), _ensure_i64(labels2),
        self_pairing, break_noise_clusters, noise_label1, noise_label2,
    )


def bcubed(labels1, labels2, *, self_pairing=False, break_noise_clusters=False,
           noise_label1=None, noise_label2=None):
    """BCubed precision, recall and F1."""
    return _fuel.bcubed(
        _ensure_i64(labels1), _ensure_i64(labels2),
        self_pairing, break_noise_clusters, noise_label1, noise_label2,
    )


def set_matching_purity(labels1, labels2, *, self_pairing=False, break_noise_clusters=False,
                        noise_label1=None, noise_label2=None):
    """Set-matching purity and inverse purity (F-measures)."""
    return _fuel.set_matching_purity(
        _ensure_i64(labels1), _ensure_i64(labels2),
        self_pairing, break_noise_clusters, noise_label1, noise_label2,
    )


def maximum_matching_accuracy(labels1, labels2, *, self_pairing=False,
                               break_noise_clusters=False, noise_label1=None,
                               noise_label2=None):
    """Maximum matching accuracy (Hungarian assignment)."""
    return _fuel.maximum_matching_accuracy(
        _ensure_i64(labels1), _ensure_i64(labels2),
        self_pairing, break_noise_clusters, noise_label1, noise_label2,
    )


def pair_sets_index(labels1, labels2, *, self_pairing=False, break_noise_clusters=False,
                    noise_label1=None, noise_label2=None):
    """Pair-sets index (simplified PSI and PSI)."""
    return _fuel.pair_sets_index(
        _ensure_i64(labels1), _ensure_i64(labels2),
        self_pairing, break_noise_clusters, noise_label1, noise_label2,
    )


def evaluate_clustering(labels1, labels2, *, self_pairing=False, break_noise_clusters=False,
                        noise_label1=None, noise_label2=None):
    """
    Compute all external cluster evaluation measures at once.

    Returns a dict with keys 'pair_counting', 'entropy', 'bcubed',
    'set_matching_purity', 'maximum_matching_accuracy', 'pair_sets_index'.
    """
    l1 = _ensure_i64(labels1)
    l2 = _ensure_i64(labels2)
    kwargs = dict(self_pairing=self_pairing, break_noise_clusters=break_noise_clusters,
                  noise_label1=noise_label1, noise_label2=noise_label2)
    return {
        'pair_counting': _fuel.pair_counting(l1, l2, self_pairing, break_noise_clusters,
                                             noise_label1, noise_label2),
        'entropy': _fuel.entropy_measures(l1, l2, self_pairing, break_noise_clusters,
                                          noise_label1, noise_label2),
        'bcubed': _fuel.bcubed(l1, l2, self_pairing, break_noise_clusters,
                               noise_label1, noise_label2),
        'set_matching_purity': _fuel.set_matching_purity(l1, l2, self_pairing,
                                                         break_noise_clusters,
                                                         noise_label1, noise_label2),
        'maximum_matching_accuracy': _fuel.maximum_matching_accuracy(
            l1, l2, self_pairing, break_noise_clusters, noise_label1, noise_label2),
        'pair_sets_index': _fuel.pair_sets_index(l1, l2, self_pairing, break_noise_clusters,
                                                 noise_label1, noise_label2),
    }


# ---------------------------------------------------------------------------
# Internal measures (require data + single label assignment)
# ---------------------------------------------------------------------------

def simplified_silhouette(data, labels, *, noise_label=None, noise_handling='ignore',
                           penalize=False):
    """
    Simplified silhouette score (centroid-based approximation).

    Returns a SilhouetteResult with 'mean', 'stddev', and 'values' (per-point array).
    """
    r = _fuel.simplified_silhouette_score(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling, penalize,
    )
    return SilhouetteResult(r['mean'], r['stddev'], r['values'])


def silhouette(data, labels, *, noise_label=None, noise_handling='ignore', penalize=False):
    """
    Full silhouette score (pairwise distances).

    Returns a SilhouetteResult with 'mean', 'stddev', and 'values' (per-point array).
    """
    r = _fuel.silhouette_score(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling, penalize,
    )
    return SilhouetteResult(r['mean'], r['stddev'], r['values'])


def davies_bouldin(data, labels, *, noise_label=None, noise_handling='ignore', p=1.0):
    """Davies-Bouldin index (lower is better)."""
    return _fuel.davies_bouldin(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling, p,
    )


def calinski_harabasz(data, labels, *, noise_label=None, noise_handling='ignore',
                      penalize=False):
    """Calinski-Harabasz / variance-ratio criterion (higher is better)."""
    return _fuel.variance_ratio(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling, penalize,
    )


def c_index(data, labels, *, noise_label=None, noise_handling='ignore'):
    """C-index (lower is better)."""
    return _fuel.c_index_score(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling,
    )


def concordance(data, labels, *, noise_label=None, noise_handling='ignore'):
    """
    Concordant-pairs gamma and tau statistics.

    Returns a dict with 'gamma' and 'tau'.
    """
    return _fuel.concordance(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling,
    )


def cluster_radius(data, labels, *, noise_label=None, noise_handling='ignore'):
    """
    Cluster radius statistics.

    Returns a dict with 'weighted' and 'unweighted' average radii.
    """
    return _fuel.cluster_radius_stats(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling,
    )


def neighbor_consistency(data, labels, k):
    """
    Neighbor consistency based on k nearest neighbors.

    Returns a dict with 'average', 'full', 'per_element_average',
    'per_element_full'.
    """
    return _fuel.neighbor_consistency(
        _ensure_f64(data), _ensure_i64(labels), k,
    )


def pbm_index(data, labels, *, noise_label=None, noise_handling='ignore'):
    """PBM index (higher is better)."""
    return _fuel.pbm(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling,
    )


def dbcv(data, labels, *, noise_label=None):
    """
    Density-based clustering validation (DBCV).

    Noise points are always merged (MergeNoise strategy).
    """
    return _fuel.dbcv_score(
        _ensure_f64(data), _ensure_i64(labels), noise_label,
    )


def squared_errors(data, labels, *, noise_label=None, noise_handling='ignore'):
    """
    Within-cluster squared error statistics (SSE/RMSD).

    Returns a dict with 'mean', 'sum_of_squares', 'rmsd'.
    """
    return _fuel.squared_error_stats(
        _ensure_f64(data), _ensure_i64(labels), noise_label, noise_handling,
    )


# ---------------------------------------------------------------------------
# Cophenetic measures (dendrograms)
# ---------------------------------------------------------------------------

def cophenetic_distances(linkage):
    """
    Compute the condensed cophenetic distance vector from a scipy linkage matrix.

    `linkage` must be an (n-1, 4) array (as returned by scipy or
    `hierarchical(...).to_scipy_linkage()`).

    Returns a 1-D float64 array of length n*(n-1)/2.
    """
    return _fuel.cophenetic_distance_vector(_ensure_f64(linkage))


def cophenetic_correlation(linkage1, linkage2):
    """
    Pearson correlation between the cophenetic distances of two dendrograms.

    Both arguments must be scipy-style linkage matrices ((n-1, 4) float64).
    Returns a scalar in [-1, 1].
    """
    return _fuel.cophenetic_corr(_ensure_f64(linkage1), _ensure_f64(linkage2))
