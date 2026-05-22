![Fuel Logo](doc/fuel-logo.png)

# Fuel - Fast Unsupervised Learning

Fuel is a Rust library implementing a broad collection of classic and research
unsupervised learning algorithms.  It targets high runtime performance and
supports multiple algorithm variants for comparison - useful for benchmarking
and academic research.

Python bindings are provided via [PyO3](https://pyo3.rs) and
[maturin](https://maturin.rs).

## Algorithms

### Clustering

| Family | Variants |
|---|---|
| **k-means** | Standard (Lloyd), Hamerly, Elkan, simplified Hamerly, simplified Elkan, Exponion, Shallot, Hartigan-Wong, MacQueen, trimmed |
| **k-medians** | standard iterative |
| **k-geometric-medians** | iterative, simplified Hamerly, kGeometric |
| **k-medoids** | PAM, FastPAM, FasterPAM, Alternating (k-means style) |
| **silhouette clustering** | PAMSil, PAMMedSil, FastMSC, DynMSC for automatic k |
| **Hierarchical (agglomerative)** | AGNES, Anderberg, Muellner, NN-chain; set-based variants; geometric NN-chain; SLINK; CLINK; search-based single-link (VP/KD/cover trees) |
| **DBSCAN** | standard, parallel |
| **OPTICS** | ordering + reachability |
| **HDBSCAN** | Prim; SLINK; search-tree-accelerated variants |
| **Spherical k-means** | Lloyd, Elkan, Hamerly, simplified Elkan, simplified Hamerly |
| **EM / Gaussian mixture** | full, spherical and diagonal covariance; von Mises-Fisher for spherical data |
| **Fuzzy c-means** | standard soft-k-means variant |

Hierarchical linkages (standard variants): single, complete, average/group_average,
weighted_average, centroid, median, Ward, minimum sum of squares, minimum
variance, minimum variance increase.

Additional linkages for set-based variants: minimax, Hausdorff, medoid, minimum
sum, minimum sum increase (HACAM).

### Outlier Detection

ABOD, FastABOD, LB-ABOD, ALOCI, LOCI, COP, DB-outlier, DWOF,
FlexibleLOF, INFLO, ISOS, Isolation Forest,
kNN-outlier, weighted kNN, KNNDD, kNN-SOS,
LDF, LDOF, LOF, LoOP, SimplifiedLOF, ODIN, SOS, IDOS, and more.

Baselines: zero, distance from center, distance from origin

### Nearest-Neighbor Search

VP-tree, KD-tree, Cover tree, linear scan.  Supports kNN, range search,
and pairwise-distance computation with a variety of distance metrics.

Optimized AVX2 kernels are used for fast and accurate Euclidean distance
computations.

### Intrinsic Dimensionality

Several estimators (MLE, TLE, GED, Zipf, ABID, RABID, ...).

## Python bindings (PyO3 + maturin)

Build and install the Python package in your current environment:

```bash
maturin develop --release
```

or install the packaged version using
```sh
pip install pyfuel
```


### Quick start

```python
import numpy as np
from fuel.cluster import kmeans, hierarchical, dbscan
from fuel.outlier import local_outlier_factor
from fuel import search

rng = np.random.default_rng(0)
data = np.vstack([
    rng.normal([0, 0], 0.5, (100, 2)),
    rng.normal([4, 0], 0.5, (100, 2)),
    rng.normal([2, 3], 0.5, (100, 2)),
]).astype(np.float32)

# k-means
result = kmeans(data, k=3)
print(f"k-means: {result.n_iter} iterations, inertia={result.inertia:.2f}")

# Hierarchical clustering
h = hierarchical(data, variant='nn_chain', linkage='ward')
labels = h.cut_by_number_of_clusters(3)
print(f"hierarchical: {labels.max() + 1} clusters")

# DBSCAN
labels = dbscan(data, eps=0.8, min_points=5)
print(f"DBSCAN: {labels.max() + 1} clusters, noise={np.sum(labels == -1)}")

# LOF outlier scores
result = local_outlier_factor(data, k=10)
print(f"LOF: top-5 outlier indices = {np.argsort(result.scores)[-5:]}")

# kNN search
index = search.SearchIndex(data, distance='euclidean')
neighbors, distances = index.knn(data[:5], k=3)
print(f"kNN: first query neighbors = {neighbors[0]}")
```

### Precomputed distances

All stored-matrix hierarchical variants, k-medoids, and related algorithms
accept a precomputed distance matrix via `distance='precomputed'`.  Both a
square matrix of shape `(n, n)` and a condensed lower-triangle vector of
length `n*(n-1)/2` are accepted:

```python
from scipy.spatial.distance import pdist, squareform
from fuel.cluster import hierarchical, kmedoids

D_square = squareform(pdist(data))
D_condensed = pdist(data)           # 1-D condensed vector

h = hierarchical(D_square, variant='agnes', linkage='ward',
                 distance='precomputed')
result = kmedoids(D_condensed, k=3, distance='precomputed')
```

## See also

- `examples/` - runnable scripts demonstrating each algorithm family
- `benchmark/` - Rust criterion benchmarks
- `comparison/` - Python scripts comparing against scikit-learn / scipy

