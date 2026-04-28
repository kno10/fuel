import numpy as _np


def _f32(data):
    return data.dtype == _np.float32


def _ensure_float(data):
    """Return data as float64 if it is not already float32 or float64."""
    if data.dtype not in (_np.float32, _np.float64):
        return data.astype(_np.float64)
    return data


def _call(f32_fn, f64_fn, data, *args, **kwargs):
    data = _ensure_float(data)
    if _f32(data):
        return f32_fn(data, *args, **kwargs)
    return f64_fn(data, *args, **kwargs)


def _f32_sparse(data):
    """Detect f32 in a scipy sparse matrix by inspecting its .data array."""
    try:
        return data.data.dtype == _np.float32
    except AttributeError:
        return False
