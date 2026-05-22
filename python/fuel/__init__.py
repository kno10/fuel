try:
    from . import _fuel
except Exception as e:
    raise ImportError("Cannot load Rust core library.") from e
from . import cluster
from . import evaluation
from . import outlier
from . import search
from .search import pairwise_distances

get_rayon_parallelism = _fuel.get_rayon_parallelism

__all__ = [
    "get_rayon_parallelism",
    "pairwise_distances",
    "cluster",
    "evaluation",
    "outlier",
    "search",
]
