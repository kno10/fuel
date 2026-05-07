# Python API

Install the package with:

```bash
maturin develop --release
```

All functions are in the `fuel` package under sub-modules:
`fuel.cluster`, `fuel.outlier`, `fuel.evaluation.cluster`, `fuel.evaluation.outlier`, and `fuel.search`.

---

## Distance functions

Most functions accept an optional `distance` keyword argument.
Accepted names (case-insensitive):

| Name | Aliases | Formula |
|------|---------|---------|
| `euclidean` | `l2` | $\sqrt{\sum_{i=1}^d (x_i-y_i)^2}$ |
| `sqeuclidean` | `squared_euclidean` | $\sum_{i=1}^d (x_i-y_i)^2$ |
| `manhattan` | `l1`, `cityblock` | $\sum_{i=1}^d \lvert x_i-y_i\rvert$ |
| `chebyshev` | `linf`, `chessboard` | $\max_{1\le i\le d} \lvert x_i-y_i\rvert$ |
| `cosine` | | $1-\frac{x\cdot y}{\|x\|\,\|y\|}$ |
| `arccosine` | `angular` | $\arccos\left(\frac{x\cdot y}{\|x\|\,\|y\|}\right)$ |
| `canberra` | | $\sum_{i=1}^d \frac{\lvert x_i-y_i\rvert}{\lvert x_i\rvert+\lvert y_i\rvert}$ |
| `braycurtis` | `bray_curtis` | $\frac{\sum_{i=1}^d \lvert x_i-y_i\rvert}{\sum_{i=1}^d \lvert x_i+y_i\rvert}$ |
| `hellinger` | | $\frac{1}{\sqrt{2}}\sqrt{\sum_{i=1}^d (\sqrt{x_i}-\sqrt{y_i})^2}$ |
| `clark` | | $\sqrt{\sum_{i=1}^d \left(\frac{x_i-y_i}{x_i+y_i}\right)^2}$ |
| `chi` | | $\sqrt{\sum_{i=1}^d \frac{(x_i-y_i)^2}{x_i+y_i}}$ |
| `chi_squared` | `chisquared`, `chi2` | $\sum_{i=1}^d \frac{(x_i-y_i)^2}{x_i+y_i}$ |
| `jensen_shannon` | `jensenshannon`, `js` | $\sqrt{\frac12\mathrm{KL}(x\|m)+\frac12\mathrm{KL}(y\|m)},\;m=\frac{x+y}{2}$ |
| `jeffrey` | `jeffreys` | $\sum_{i=1}^d (x_i-y_i)(\ln x_i - \ln y_i)$ |
| `histogram_intersection` | `intersection` | $1-\sum_{i=1}^d \min(x_i,y_i)$ |


Additional distance functions are already available in the Rust API!

The default is `euclidean` unless stated otherwise.

---

## Data types

Inputs are accepted as any numeric NumPy array.
Non-float arrays are automatically cast to `float64`.
`float32` arrays are processed with single-precision arithmetic throughout.

Other numeric types are currently not supported by the Python wrapper -- use the Rust API if you need extended support.

---

## Search API

The search API lives in `fuel.search`.

```python
import fuel.search as search
```

### k-Nearest-Neighbors search

Finding the nearest neighbors of a point can be performed as follows:

```python
indices, distances = search.knn_search(
    data, query, k,
    *,
    exclude_self=None,
    distance='euclidean',
    tree='vp',
    seed=None,
)
```

Find the `k` nearest neighbors for every point in `query` against `data`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | ndarray `(n, d)` | Input data. |
| `query` | ndarray `(m, d)` | Query points. |
| `k` | int | Number of neighbors per query point. |
| `exclude_self` | bool or None | Exclude the query point itself (distance 0). Default `None` selects ``True`` only when ``query`` refers to the same underlying array as ``data``. If ``False``, query points may be included in the results. If ``query`` is a separate array, ``exclude_self`` defaults to ``False`` and explicit ``True`` is not supported. |
| `distance` | str | Distance function. Default `'euclidean'`. Supported names: `euclidean`/`l2`, `sqeuclidean`/`squared_euclidean`, `manhattan`/`l1`/`cityblock`, `chebyshev`/`linf`/`chessboard`, `cosine`, `arccosine`/`angular`, `canberra`, `braycurtis`/`bray_curtis`, `hellinger`, `clark`, `chi`, `chi_squared`/`chisquared`/`chi2`, `jensen_shannon`/`jensenshannon`/`js`, `jeffrey`/`jeffreys`, `histogram_intersection`/`intersection`. When `tree='kd'` only `euclidean`, `sqeuclidean`, and `manhattan` are supported. |
| `tree` | `{'auto', 'vp', 'kd', 'cover'}` | Index structure. `'auto'` chooses `'kd'` for low-dimensional, coordinate-based distances and otherwise `'vp'`. `'vp'` and `'cover'` support all distances, but are exact only for metric distances |
| `seed` | int or None | RNG seed for tree construction (`'vp'` and `'cover'` only). |

**Returns** `(indices, distances)`:
- `indices` - `int64` array of shape `(m, k)`. Entries are `-1` where fewer than `k` neighbors exist.
- `distances` - float array of shape `(m, k)`. Entries are `inf` where fewer than `k` neighbors exist.

---

### Radius search

Radius or range search (sometimes also called window search, but not to be confused with a hyperbox search) can be performed as follows:

```python
indices, distances = search.range_search(
    data, query, radius,
    *,
    exclude_self=None,
    distance='euclidean',
    tree='vp',
    seed=None,
)
```

Find all neighbors within `radius` of each point in `query` against `data`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | ndarray `(n, d)` | Input data. |
| `query` | ndarray `(m, d)` | Query points. |
| `radius` | float | Search radius (inclusive). |
| `exclude_self` | bool or None | Exclude the query point itself. The default is ``True`` only when ``query`` refers to the same underlying array as ``data``. If ``False``, query points may be included in the results, the only valid value if ``query`` and ``data`` are not the same. |
| `distance` | str | Distance function. Default `'euclidean'`. |
| `tree` | `{'auto', 'vp', 'kd', 'cover'}` | Index structure. Default `'vp'`. `'auto'` chooses `'vp'` for the default radius search path. `'kd'` supports coordinate-based radius search for suitable distances. `'vp'` and `'cover'` are exact only for metric distances. |
| `seed` | int or None | RNG seed for tree construction. |

**Returns** `(indices, distances)`:
- `indices` - list of length `m` containing 1-D ndarrays with neighbor indices for each query.
- `distances` - list of length `m` containing 1-D ndarrays with neighbor distances sorted by distance.

### Search index

Build a persistent search index once and use it for repeated queries:

```python
index = search.SearchIndex(data, distance='euclidean', tree='auto', seed=None)
```

Note: modifying the data array after building the index is not supported,
because the data is not copied. This can yield incorrect results.

**Parameters**

- `data` : ndarray `(n, d)`
- `distance` : str, optional
- `tree` : `{'auto', 'vp', 'kd', 'cover', 'linear'}`
  - `'auto'` chooses `'kd'` for Euclidean-like distances on low-dimensional inputs, otherwise `'vp'`.
  - `'vp'` builds a persistent VP-tree index.
  - `'cover'` builds a persistent cover-tree index.
  - `'kd'` uses the existing KD-tree-based search code path for repeated `knn` queries.
  - `'linear'` builds a brute-force linear-scan searcher.
- `seed` : int or None, optional
  - RNG seed for VP-tree or cover-tree construction.

**Returns**

- `SearchIndex`

The returned object supports:

- `index.knn(k, exclude_self=False)`
- `index.radius_search(radius, exclude_self=False)`

For `tree='kd'`, only `knn` is supported; `radius_search` raises an error.

Both VP-tree and cover-tree are exact only for metric distances.

---

## Clustering

```python
import fuel.cluster as cluster
```

### K-Means Clustering

K-means clustering minimizes the sum of squared errors (squared Euclidean distance).

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kmeans(
    data, *, k,
    variant='simplified_elkan',
    max_iter=300,
    tol=0,
    seed=None,
    init=None,
)
```

The input must be a dense ndarray.

**Variants:** `lloyd`, `lloyd_blas`, `lloyd_naive`, `elkan`, `simp_elkan`
(`simplified_elkan`), `hamerly`, `simp_hamerly` (`simplified_hamerly`),
`exponion`, `shallot`, `hartigan_wong`, `hartigan_wong_quick`, `macqueen`.

**`init`:** `'random'`, `'first'`, `'kmeans++'`, `'kgeometric++'`, or a
`(k, d)` ndarray of fixed initial centers.

---

### K-Medians clustering

K-medians clustering uses the median on each axis instead of the mean,
which minimizes Manhattan distances.

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kmedians(
    data, *, k, max_iter=300, tol=0, seed=None, init=None,
)
```

K-medians clustering (Manhattan distance).

---

### K-Geometric Medians clustering

This uses the geometric median to optimize Euclidean distances.

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kgeometric(
    data, *, k, steps, variant='default', max_iter=300, tol=1e-4, seed=None, init=None,
)
```

K-geometric-means. `steps` controls the number of geometric update sub-steps.
**Variants:** `'default'`, `'sh'` (Hamerly-accelerated).

---

#### GMedians clustering

This is an alternative approach also using geometric medians.

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kgmedians(
    data, *, k, gamma, alpha, max_iter=300, tol=1e-4, seed=None, init=None,
)
```

Generalised k-medians with parameters `gamma` and `alpha`.

---

### K-Harmonic means clustering

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kharmonic(
    data, *, k, p, max_iter=300, tol=1e-4, seed=None, init=None,
)
```

K-harmonic means with harmonic power `p`.

---

### Trimmed k-means Clustering

Also known as k-means\-\-.

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.tkmeans(
    data, *, k, alpha, max_iter=300, tol=0, seed=None, init=None,
)
```

Trimmed k-means. `alpha` is the trimming proportion in `[0, 1)`.

---

### Fuzzy c-means clustering

```python
centers, membership, assignments, iterations, loss = cluster.fuzzycmeans(
    data, *, k, m, max_iter=300, tol=1e-4, seed=None, init=None,
)
```

Fuzzy c-means (Lloyd update). `m` is the fuzziness exponent (>1).
Returns membership matrix of shape `(n, k)` in addition to hard assignments.

---

### Spherical k-Means clustering

Spherical k-means minimizes the angle between the data points and the cluster
direction.

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.spherical_kmeans(
    data, *, k,
    variant='simp_elkan',
    max_iter=300,
    tol=0,
    seed=None,
    init=None,
)
```

Spherical k-means (cosine distance). Accepts dense ndarray or CSR sparse matrix.

**Variants:** `lloyd`, `elkan`, `simp_elkan`, `hamerly`, `simp_hamerly`.

---

### Gaussian Mixture Modeling (EM Clustering)

Each cluster is modeled using a multivariate Gaussian distributions.
Three different cluster models are supported (spherical, axis-aligned aka. diagonal covariance matrix, and a fully multivariate model that allows rotated Gaussians). Three variants with differnt numerical behavior are supported, but its usually fine to stick to the default approach.
Prior can be used to use a maximum-a-posteriori approach, where the prior is based on the overall data distribution.

```python
weights, means, variances, assignments, responsibilities, n_iter, log_likelihood = cluster.em(
    data, k,
    *,
    model='diagonal',
    variant='default',
    delta=1e-5,
    miniter=10,
    maxiter=200,
    hard=False,
    prior=0.0,
    return_soft=False,
    min_log_likelihood=-1e300,
    noise_ratio=0.0,
    seed=None,
)
```

Gaussian mixture model EM.

**`model`:** `'diagonal'`, `'spherical'`, `'multivariate'`.
**`variant`:** `'default'`, `'textbook'`, `'two_pass'`.

For `'multivariate'`, `variances` is a covariance matrix array of shape `(k, d, d)`;
for `'diagonal'` it is shape `(k, d)`, for `'spherical'` shape `(k,)`.

When `return_soft=True`, `responsibilities` is the full `(n, k)` soft-assignment matrix;
otherwise it is `None`.

---

### Clustering with von-Mises-Fisher distributions

This is an expectation-maximization approach for points on the sphere, e.g., on text data. In contrast to spherical k-means, it is a soft clustering approach, and clusters can have different diameters.

```python
weights, means, kappas, assignments, responsibilities, n_iter, log_likelihood = cluster.von_mises_fisher(
    data, k,
    *,
    delta=1e-5,
    miniter=10,
    maxiter=200,
    hard=False,
    prior=0.0,
    return_soft=False,
    min_log_likelihood=-1e300,
    noise_ratio=0.0,
    init_kappa=1.0,
    seed=None,
)
```

Von Mises-Fisher mixture model EM. Accepts a CSR sparse matrix.

---

### Hierarchical clustering

Fuel supports a wide range of hierarchical clustering approaches.
Not every linkage is supported by every optimization strategy, this is inherent to the optimizations used for some algorithms.

```python
result = cluster.hierarchical(
    data,
    variant='auto',
    linkage='ward',
    *,
    distance=None,
    sample_size=None,
    slack=None,
    seed=None,
)
```

Hierarchical agglomerative clustering. Returns a `MergeHistory` object.

**Variants and their linkage sets:**

| Variant | Description | Supported linkages |
|---------|-------------|--------------------|
| `agnes` | Standard O(n^3) AGNES | All standard |
| `anderberg` | Anderberg's update formula | All standard |
| `muellner` | Muellner optimised | All standard |
| `nn_chain` | NN-chain, O(n^2) | All standard |
| `set_agnes`, `set_anderberg`, `set_muellner`, `set_nn_chain` | Set-based; include minimax / hausdorff / medoid | Extended |
| `geometric_nn_chain` | Euclidean geometry, no distance parameter | Geometric only |
| `incremental_nn_chain` | Incremental search; requires `sample_size` | Geometric only |
| `slink` | Sibson SLINK, O(n^2) memory | Fixed: `single` |
| `clink` | Defays CLINK | Fixed: `complete` |
| `boruvka_searchers_single_link` | Boruvka+searchers; requires `sample_size` | Fixed: `single` |
| `heap_of_searchers_single_link` | Heap-of-searchers; requires `sample_size` | Fixed: `single` |
| `restarting_search_single_link` | Restarting search; requires `sample_size` | Fixed: `single` |
| `buffered_search_single_link` | Buffered; requires `sample_size`, `slack` | Fixed: `single` |
| `lazy_buffered_search_single_link` | Lazy buffered; requires `sample_size`, `slack` | Fixed: `single` |

**Standard linkages:** `single`, `complete`, `average` (`group_average`,
`weighted_average`), `centroid`, `median`, `ward` (`missq`), `minimum_sum_squares`
(`mnssq`), `minimum_variance_increase` (`mivar`), `minimum_variance` (`mnvar`).

**Extended linkages** (set-based variants only): all standard plus `minimax`,
`hausdorff`, `medoid`, `minimum_sum` (`mnsum`), `minimum_sum_increase` (`misum`).

**Geometric linkages** (geometric/incremental variants): `average`, `centroid`,
`ward`, `missq`, `mnssq`, `mivar`, `mnvar`.

**`MergeHistory` methods:**

```python
labels = result.cut_by_number_of_clusters(k) # int64 array
labels = result.cut_by_height(height)        # int64 array
Z      = result.to_scipy_linkage()           # (n-1, 4) float array
```

---

### HDBSCAN clustering

```python
result = cluster.hdbscan(
    data, min_points,
    variant='hdbscan_prim',
    *,
    distance=None,
    sample_size=None,
    slack=None,
    seed=None,
)
```

HDBSCAN hierarchy construction. Returns an `HdbscanHierarchy` object.

**Variants:**

| Variant | Description |
|---------|-------------|
| `hdbscan_prim` | Prim's MST on mutual reachability, O(n^2) |
| `slink_hdbscan` | SLINK-style, O(n^2) |
| `heap_of_searchers_hdbscan` | Tree-accelerated; requires `sample_size` |
| `restarting_search_hdbscan` | Tree-accelerated; requires `sample_size` |
| `boruvka_searchers_hdbscan` | Tree-accelerated; requires `sample_size` |
| `buffered_search_hdbscan` | Tree-accelerated; requires `sample_size`, `slack` |
| `lazy_buffered_search_hdbscan` | Tree-accelerated; requires `sample_size`, `slack` |

**`HdbscanHierarchy` methods:**

```python
core_dists = result.core_distances()                              # 1-D float array
Z          = result.to_scipy_linkage()                            # (n-1, 4) float array
labels     = result.extract_clusters_with_noise(num_clusters, min_cluster_size)
info       = result.extract_simplified(min_cluster_size)          # dict
info       = result.extract_hdbscan(min_cluster_size, hierarchical) # dict
```

---

### DBSCAN

Density-based clustering with noise.

```python
labels = cluster.dbscan(
    data,
    eps,
    min_points,
    *,
    distance="euclidean",
    variant="dbscan",
    seed=None,
)
```

DBSCAN. Returns `int64` labels; `-1` indicates noise.

Use `variant='parallel'` to select the parallel DBSCAN implementation.

---

### OPTICS Clustering

A successor to DBSCAN and precursor to HDBSCAN. Typically, HDBSCAN* is to be preferred.

```python
result = cluster.optics(data, max_eps, min_points, *, distance="euclidean", seed=None)
```

OPTICS ordering and reachability. Returns an `OpticsResult` object.

**`OpticsResult` methods:**

```python
ordering      = result.ordering()       # processing order, int64 array
reachability  = result.reachability()   # reachability distances, float array
core_dist     = result.core_distance()  # core distances (NaN if not core), float array
predecessor   = result.predecessor()    # predecessor indices (-1 if none), int64 array
labels        = result.labels()         # DBSCAN-style labels from initial run, int64 array
labels        = result.extract_xi(xi, min_points)  # Xi-based extraction, int64 array
```

---

## Outlier Detection

```python
import fuel.outlier as outlier
```

All outlier functions return a tuple `(scores, metadata)` where `scores` is a
1-D float array of length `n` and `metadata` is a dict containing:

| Key | Description |
|-----|-------------|
| `label` | Short name of the method |
| `ascending` | `True` if higher score = more outlying |
| `baseline` | Expected baseline score |
| `minimum` | Observed minimum score |
| `maximum` | Observed maximum score |
| `theoretical_minimum` | Theoretical minimum (may be `NaN`) |
| `theoretical_maximum` | Theoretical maximum (may be `NaN`) |

The two baseline methods (`zero`, `random`) follow the same calling convention
but the return format matches the standard tuple.

The Rust backend of fuel contains a binary that can compute a parameter
sweep over k for all kNN-based outlier detection methods *much* faster than
iterating over this from Python: it precomputes the kNN for the maximum k
just *once*, then runs the outlier detectors using this information.
It can be built using the command
```sh
cargo build --release --bin compute_knn_outlier_scores --features parallel,io
```

### Angle-based

| Function | Parameters | Notes |
|----------|-----------|-------|
| `angle_based_outlier_detection(data, *, kernel, distance)` | `kernel`: `'poly2'` (default), `'poly3'`, `'linear'` | ABOD |
| `fast_angle_based_outlier_detection(data, k, *, kernel, seed, distance)` | | FastABOD |
| `locality_based_abod(data, k, l, *, distance)` | | LB-ABOD |

### Correlation / subspace

| Function | Parameters | Notes |
|----------|-----------|-------|
| `approximate_local_correlation_integral(data, nmin, alpha, g, *, seed, distance)` | `nmin`: minimum neighborhood size, `alpha`: smoothing parameter, `g`: kernel exponent | ALOCI |
| `local_correlation_integral(data, rmax, nmin, alpha, *, seed, distance)` | `rmax`: radius threshold, `nmin`: minimum neighborhood size, `alpha`: smoothing parameter | LOCI |
| `correlation_outlier_probabilities(data, k, expect, dist, *, seed, distance)` | `k`: neighbors, `expect`: expected neighbor count, `dist`: `'chi2'` or `'gamma'` | COP |
| `local_intrinsic_dimensionality(data, k, *, estimator, seed, distance)` | `k`: neighbors, `estimator`: LID estimator name | LID-based |
| `intrinsic_dimensionality_outlier_score(data, k_c, k_r, *, estimator, seed, distance)` | `k_c`: reference neighbors, `k_r`: reachability neighbors, `estimator`: LID estimator | IDOS |
| `subspace_outlier_degree(data, k, alpha, *, seed, distance)` | `k`: neighbors, `alpha`: subspace balance parameter | SOD |

### Distance / density based

The function signatures below show the available parameters. Common arguments are:
- `k`: number of nearest neighbors used for the score.
- `distance`: metric name, default `euclidean`.
- `seed`: RNG seed for repeatable results.

| Function | Notes |
|----------|-------|
| `k_nearest_neighbors_outlier(data, k, *, seed, distance)` | kNN distance outlier |
| `k_nearest_neighbors_distance_deviation(data, k, *, seed, distance)` | kNNDD |
| `k_nearest_neighbors_sos(data, k, *, seed, distance)` | kNN-SOS |
| `weighted_knn(data, k, *, seed, distance)` | Weighted kNN |
| `local_outlier_factor(data, k, *, seed, distance)` | LOF |
| `simplified_lof(data, k, *, seed, distance)` | Simplified LOF |
| `flexible_lof(data, krefer, kreach, *, seed, distance)` | Flexible LOF, `krefer` reference set size, `kreach` reachability count |
| `local_density_outlier_factor(data, k, *, seed, distance)` | LDOF |
| `local_outlier_probabilities(data, k, m, *, seed, distance)` | LoOP, `m` smoothing parameter |
| `dynamic_window_outlier_factor(data, k, delta, *, seed, distance)` | DWOF, `delta` window size |
| `local_density_factor(data, k, h, c, kernel, *, seed, distance)` | LDF, `h` bandwidth, `c` kernel parameter |
| `simple_kernel_density_lof(data, k, h, kernel, *, seed, distance)` | KDEOS, `h` bandwidth, `kernel` name |
| `stochastic_outlier_selection(data, perplexity, *, seed, distance)` | SOS, `perplexity` effective neighbor count |
| `outlier_detection_independence_neighbor(data, k, *, seed, distance)` | ODIN |
| `local_isolation_coefficient(data, k, *, seed, distance)` | LIC |
| `influence_outlier(data, k, m, *, seed, distance)` | `m` influence exponent |
| `variance_of_volume(data, k, *, seed, distance)` | VOV |
| `connectivity_outlier_factor(data, k, *, seed, distance)` | COF |

### Center / distance from reference

| Function | Parameters |
|----------|-----------|
| `distance_from_center(data, *, distance)` | Distance to centroid |
| `distance_from_origin(data, *, distance)` | Distance to origin |

### DB-outlier

| Function | Parameters |
|----------|-----------|
| `db_outlier_score(data, d, *, seed, distance)` | |
| `db_outlier_detection(data, d, p, *, seed, distance)` | |

### Forest-based

| Function | Parameters |
|----------|-----------|
| `isolation_forest(data, num_trees, subsample_size, *, seed)` | No `distance` parameter |

### Baselines

| Function | Parameters |
|----------|-----------|
| `zero(data)` | Returns zero score for every point |
| `random(data, *, seed)` | Returns uniform random scores |

### LID estimators (`estimator` keyword)

Used by `local_intrinsic_dimensionality` and `intrinsic_dimensionality_outlier_score`:
`'hill'`, `'aggregated_hill'`, `'ged'`, `'mom'`, `'pbm_lid'`, `'alid'`, `'abid'`,
`'rabid'`, `'rv'`, `'zipf'`, `'tightlid'`, `'lmom'`.

### Kernel names (`kernel` keyword)

Used by `local_density_factor` and `simple_kernel_density_lof`:
`'uniform'`, `'triangular'`, `'epanechnikov'`, `'biweight'`, `'triweight'`,
`'cosine'`, `'gaussian'`.

---

## Evaluation methods

```python
import fuel.evaluation.cluster
import fuel.evaluation.outlier
```

### External clustering measures

All functions below compare two label assignments. Cluster labels are `int64` arrays.
Noise points can be handled with `noise_label` / `noise_label1` / `noise_label2` and
`break_noise_clusters`.

| Function | Returns |
|----------|---------|
| `pair_counting(labels1, labels2, *, self_pairing, break_noise_clusters, noise_label1, noise_label2)` | dict: F1, precision, recall, ARI, Jaccard, ... |
| `entropy_measures(labels1, labels2, ...)` | dict: MI, NMI variants, VI, conditional entropy, ... |
| `bcubed(labels1, labels2, ...)` | dict: BCubed precision, recall, F1 |
| `set_matching_purity(labels1, labels2, ...)` | dict: purity, inverse purity, F-measure |
| `maximum_matching_accuracy(labels1, labels2, ...)` | dict: MMA (Hungarian) |
| `pair_sets_index(labels1, labels2, ...)` | dict: simplified PSI and PSI |
| `evaluate_clustering(labels1, labels2, ...)` | dict of all the above |

### Internal clustering measures

Data is a `float64` `(n, d)` array; labels are `int64`.

| Function | Returns | Notes |
|----------|---------|-------|
| `simplified_silhouette(data, labels, *, noise_label, noise_handling, penalize)` | dict: mean, stddev, values | Centroid-based |
| `silhouette(data, labels, *, noise_label, noise_handling, penalize)` | dict: mean, stddev, values | Pairwise |
| `davies_bouldin(data, labels, *, noise_label, noise_handling, p)` | float | Lower is better |
| `calinski_harabasz(data, labels, *, noise_label, noise_handling, penalize)` | float | Higher is better |
| `c_index(data, labels, *, noise_label, noise_handling)` | float | Lower is better |
| `concordance(data, labels, *, noise_label, noise_handling)` | dict: gamma, tau | |
| `cluster_radius(data, labels, *, noise_label, noise_handling)` | dict: weighted, unweighted | |
| `neighbor_consistency(data, labels, k)` | dict: average, full, per_element_average, per_element_full | |
| `pbm_index(data, labels, *, noise_label, noise_handling)` | float | Higher is better |
| `dbcv(data, labels, *, noise_label)` | float | Density-based; noise is always merged |
| `squared_errors(data, labels, *, noise_label, noise_handling)` | dict: mean, sum_of_squares, rmsd | |

`noise_handling` accepts `'ignore'` (default) or `'penalize'`.

### Dendrogram / cophenetic measures

```python
coph_vec  = evaluation.cophenetic_distances(linkage)
corr      = evaluation.cophenetic_correlation(linkage1, linkage2)
```

`linkage` must be an `(n-1, 4)` scipy-style linkage matrix (as returned by
`hierarchical(...).to_scipy_linkage()`).

### Outlier evaluation measures

Scores are `float64` arrays; labels are binary `uint8` arrays (1 = outlier).

The adjusted measures subtract an expected baseline to correct for chance performance, following ELKI-style adjusted evaluation.

| Function | Returns |
|----------|---------|
| `auroc(scores, labels)` | AUROC |
| `adjusted_auroc(scores, labels)` | Adjusted AUROC |
| `average_precision(scores, labels)` | AP |
| `adjusted_average_precision(scores, labels)` | Adjusted average precision |
| `auprc(scores, labels)` | Area under PR curve |
| `adjusted_auprc(scores, labels)` | Adjusted area under PR curve |
| `adjusted_auprgc(scores, labels)` | Adjusted area under PR-gain curve |
| `pr_curve(scores, labels)` | dict: recall, precision (1-D arrays) |
| `prg_auc(scores, labels)` | Area under PR-gain curve |
| `dcg(scores, labels)` | DCG |
| `adjusted_dcg(scores, labels)` | Adjusted DCG |
| `ndcg(scores, labels)` | NDCG |
| `maximum_f1(scores, labels)` | Max F1 across thresholds |
| `adjusted_maximum_f1(scores, labels)` | Adjusted max F1 |
| `precision_at_k(scores, labels, k)` | Precision@k |
| `r_precision(scores, labels)` | R-precision |
| `adjusted_r_precision(scores, labels)` | Adjusted R-precision |
