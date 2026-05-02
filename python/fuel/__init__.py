try:
    from . import _fuel
except:
    raise "Cannot load Rust core library."
from . import cluster
from . import evaluation
from . import outlier

get_rayon_parallellism = _fuel.get_rayon_parallellism

__all__ = ["get_rayon_parallellism", "cluster", "evaluation", "outlier"]