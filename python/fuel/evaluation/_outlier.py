import numpy as np
from .. import _fuel as _fuel


def _ensure_f64(scores):
    a = np.asarray(scores, dtype=np.float64)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return a


def _ensure_u8(labels):
    a = np.asarray(labels, dtype=np.uint8)
    if not a.flags['C_CONTIGUOUS']:
        a = np.ascontiguousarray(a)
    return a


def auroc(scores, labels):
    """Area under the ROC curve (AUROC). Tie-aware.

    Parameters
    ----------
    scores : array_like
        Score values for each sample.
    labels : array_like
        Binary labels indicating the positive class.
    """
    return _fuel.outlier_auroc(_ensure_f64(scores), _ensure_u8(labels))


def average_precision(scores, labels):
    """Average precision (AP). Tie-aware."""
    return _fuel.outlier_average_precision(_ensure_f64(scores), _ensure_u8(labels))


def auprc(scores, labels):
    """Area under the precision-recall curve. Trapezoid rule on ELKI-style PR curve."""
    return _fuel.outlier_auprc(_ensure_f64(scores), _ensure_u8(labels))


def adjusted_auroc(scores, labels):
    """Adjusted area under the ROC curve (AUROC with random baseline at 0)."""
    return _fuel.outlier_adjusted_auroc(_ensure_f64(scores), _ensure_u8(labels))


def adjusted_auprc(scores, labels):
    """Adjusted area under the precision-recall curve."""
    return _fuel.outlier_adjusted_auprc(_ensure_f64(scores), _ensure_u8(labels))


def adjusted_auprgc(scores, labels):
    """Adjusted area under the precision-recall gain curve."""
    return _fuel.outlier_adjusted_auprgc(_ensure_f64(scores), _ensure_u8(labels))


def adjusted_average_precision(scores, labels):
    """Adjusted average precision."""
    return _fuel.outlier_adjusted_average_precision(_ensure_f64(scores), _ensure_u8(labels))


def adjusted_r_precision(scores, labels):
    """Adjusted R-Precision."""
    return _fuel.outlier_adjusted_r_precision(_ensure_f64(scores), _ensure_u8(labels))


def adjusted_maximum_f1(scores, labels):
    """Adjusted maximum F1 score."""
    return _fuel.outlier_adjusted_maximum_f1(_ensure_f64(scores), _ensure_u8(labels))


def adjusted_dcg(scores, labels):
    """Adjusted DCG, computed from normalized DCG with a random baseline."""
    return _fuel.outlier_adjusted_dcg(_ensure_f64(scores), _ensure_u8(labels))


def pr_curve(scores, labels):
    """
    Precision-recall curve.

    Returns a dict with 'recall' and 'precision' as float64 arrays.
    """
    return _fuel.outlier_pr_curve(_ensure_f64(scores), _ensure_u8(labels))


def prg_auc(scores, labels):
    """Area under the precision-recall gain curve."""
    return _fuel.outlier_prg_auc(_ensure_f64(scores), _ensure_u8(labels))


def dcg(scores, labels):
    """Discounted cumulative gain. Tie-aware."""
    return _fuel.outlier_dcg(_ensure_f64(scores), _ensure_u8(labels))


def ndcg(scores, labels):
    """Normalized discounted cumulative gain. Tie-aware."""
    return _fuel.outlier_ndcg(_ensure_f64(scores), _ensure_u8(labels))


def maximum_f1(scores, labels):
    """Maximum F1 score across thresholds. Tie-aware."""
    return _fuel.outlier_maximum_f1(_ensure_f64(scores), _ensure_u8(labels))


def precision_at_k(scores, labels, k):
    """Precision at rank k. Tie-aware."""
    return _fuel.outlier_precision_at_k(_ensure_f64(scores), _ensure_u8(labels), k)


def r_precision(scores, labels):
    """R-precision (precision at k = number of positives). Tie-aware."""
    return _fuel.outlier_r_precision(_ensure_f64(scores), _ensure_u8(labels))
