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


def em(data, k, *, model='diagonal', variant='default', delta=1e-5, miniter=10,
       maxiter=200, hard=False, prior=0.0, return_soft=False,
       min_log_likelihood=-1e300, noise_ratio=0.0, seed=None):
    """
    Gaussian mixture model EM clustering.

    model : 'diagonal' (default) | 'spherical' | 'multivariate'
    variant : 'default' (default) | 'textbook' | 'two_pass' (multivariate only)

    Returns (weights, means, variances_or_covariances, assignments,
             responsibilities, n_iter, log_likelihood).
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
    return fn(data, k, delta, miniter, maxiter, hard, prior,
              return_soft, min_log_likelihood, noise_ratio, v, seed)


def von_mises_fisher(data, k, *, delta=1e-5, miniter=10, maxiter=200,
                        hard=False, prior=0.0, return_soft=False,
                        min_log_likelihood=-1e300, noise_ratio=0.0,
                        init_kappa=1.0, seed=None):
    """
    Von Mises-Fisher EM on a CSR sparse matrix.

    Returns (weights, means, kappas, assignments, responsibilities,
             n_iter, log_likelihood).
    """
    if _f32_sparse(data):
        return _fuel.von_mises_fisher_em_sparse_f32(
            data, k, delta, miniter, maxiter, hard, prior,
            return_soft, min_log_likelihood, noise_ratio, init_kappa, seed)
    return _fuel.von_mises_fisher_em_sparse_f64(
        data, k, delta, miniter, maxiter, hard, prior,
        return_soft, min_log_likelihood, noise_ratio, init_kappa, seed)
