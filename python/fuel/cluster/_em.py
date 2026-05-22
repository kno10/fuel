from .. import _fuel as _fuel
from .._dispatch import _ensure_float, _f32, _f32_sparse

_EM_MODELS = {
    'diagonal':     ('diagonal_em_f32',     'diagonal_em_f64'),
    'spherical':    ('spherical_em_f32',    'spherical_em_f64'),
    'multivariate': ('multivariate_em_f32', 'multivariate_em_f64'),
}

_EM_MODEL_VARIANTS = {
    'diagonal':     frozenset({'default', 'textbook', 'two_pass'}),
    'spherical':    frozenset({'default', 'textbook', 'two_pass'}),
    'multivariate': frozenset({'default', 'textbook', 'two_pass'}),
}


class EMResult:
    """Expectation-maximization clustering result.

    Parameters
    ----------
    weights : ndarray
        Cluster mixture weights.
    means : ndarray
        Cluster means.
    parameters : ndarray
        Component-specific parameters (variances, covariances, or kappas).
    assignments : ndarray
        Hard cluster assignments.
    responsibilities : ndarray or None
        Soft assignment matrix if `return_soft=True`, otherwise `None`.
    n_iter : int
        Number of EM iterations.
    log_likelihood : float
        Final log-likelihood.
    """

    def __init__(self, weights, means, parameters, assignments,
                 responsibilities, n_iter, log_likelihood):
        self.weights = weights
        self.means = means
        self.parameters = parameters
        self.assignments = assignments
        self.responsibilities = responsibilities
        self.n_iter = n_iter
        self.log_likelihood = log_likelihood

    def __repr__(self):
        return (
            f"EMResult(weights={self.weights}, means={self.means}, "
            f"parameters={self.parameters}, assignments={self.assignments}, "
            f"responsibilities={self.responsibilities}, n_iter={self.n_iter}, "
            f"log_likelihood={self.log_likelihood})"
        )


def _build_em_result(result):
    if isinstance(result, EMResult):
        return result
    if not isinstance(result, tuple):
        return result
    if len(result) == 7:
        weights, means, parameters, assignments, responsibilities, n_iter, log_likelihood = result
        return EMResult(weights, means, parameters, assignments,
                        responsibilities, n_iter, log_likelihood)
    return result


def em(data, *, k, model='diagonal', variant='default', tol=1e-5, min_iter=10,
       max_iter=200, hard=False, prior=0.0, return_soft=False,
       min_log_likelihood=-1e300, noise_ratio=0.0, seed=None):
    """
    Gaussian mixture model EM clustering.

    model : 'diagonal' (default) | 'spherical' | 'multivariate'
    variant : 'default' (default) | 'textbook' | 'two_pass' (multivariate only)

    Returns
    -------
    EMResult
        Named result object with `weights`, `means`, `parameters`,
        `assignments`, `responsibilities`, `n_iter`, and `log_likelihood`.
    """
    m = model.lower()
    if m not in _EM_MODELS:
        raise ValueError(
            f"unknown em model '{model}'. Valid: {sorted(_EM_MODELS)}"
        )
    v = variant.lower()
    if v not in _EM_MODEL_VARIANTS[m]:
        raise ValueError(
            f"variant '{variant}' is not valid for model '{model}'. "
            f"Valid: {sorted(_EM_MODEL_VARIANTS[m])}"
        )
    data = _ensure_float(data)
    f32_name, f64_name = _EM_MODELS[m]
    fn = getattr(_fuel, f32_name if _f32(data) else f64_name)
    return _build_em_result(fn(data, k, tol, min_iter, max_iter, hard, prior,
                                 return_soft, min_log_likelihood, noise_ratio, v, seed))


def von_mises_fisher(data, *, k, tol=1e-5, min_iter=10, max_iter=200,
                        hard=False, prior=0.0, return_soft=False,
                        min_log_likelihood=-1e300, noise_ratio=0.0,
                        init_kappa=1.0, seed=None):
    """
    Von Mises-Fisher EM on a CSR sparse matrix.

    Returns
    -------
    EMResult
        Named result object with `weights`, `means`, `parameters`,
        `assignments`, `responsibilities`, `n_iter`, and `log_likelihood`.
    """
    if _f32_sparse(data):
        return _build_em_result(_fuel.von_mises_fisher_em_sparse_f32(
            data, k, tol, min_iter, max_iter, hard, prior,
            return_soft, min_log_likelihood, noise_ratio, init_kappa, seed))
    return _build_em_result(_fuel.von_mises_fisher_em_sparse_f64(
        data, k, tol, min_iter, max_iter, hard, prior,
        return_soft, min_log_likelihood, noise_ratio, init_kappa, seed))
