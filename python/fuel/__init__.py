try:
    from . import _fuel
except:
    raise "Cannot load Rust core library."
from . import cluster
from . import outlier
