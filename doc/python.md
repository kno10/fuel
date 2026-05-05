# Python API

Install the package with:

```bash
maturin develop --release
```

All functions are in the `fuel` package under sub-modules:
`fuel.cluster`, `fuel.outlier`, `fuel.evaluation`, `fuel.search`.

---

## Distance functions

Most functions accept an optional `distance` keyword argument.
Accepted names (case-insensitive):

| Name | Aliases |
|------|---------|
| `euclidean` | `l2` |
| `sqeuclidean` | `squared_euclidean` |
| `manhattan` | `l1`, `cityblock` |
| `chebyshev` | `linf`, `chessboard` |
| `cosine` | |
| `arccosine` | `angular` |
| `canberra` | |
| `braycurtis` | `bray_curtis` |
| `hellinger` | |
| `clark` | |
| `chi` | |
| `chi_squared` | `chisquared`, `chi2` |
| `jensen_shannon` | `jensenshannon`, `js` |
| `jeffrey` | `jeffreys` |
| `histogram_intersection` | `intersection` |

The default is `euclidean` unless stated otherwise.

---

## Data types

Inputs are accepted as any numeric NumPy array.
Non-float arrays are automatically cast to `float64`.
`float32` arrays are processed with single-precision arithmetic throughout.

---

## fuel.search

```python
import fuel.search as search
```

### `search.knn_search`

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
| `tree` | `{'auto', 'vp', 'kd', 'cover'}` | Index structure. `'auto'` chooses `'kd'` for low-dimensional, coordinate-based distances and otherwise `'vp'`. `'vp'` supports all distances; `'kd'` can be faster for low-dimensional Euclidean-like data; `'cover'` uses a cover tree and supports all distances. `'vp'` and `'cover'` are exact only for metric distances. |
| `seed` | int or None | RNG seed for tree construction (`'vp'` and `'cover'` only). |

**Returns** `(indices, distances)`:
- `indices` - `int64` array of shape `(m, k)`. Entries are `-1` where fewer than `k` neighbors exist.
- `distances` - float array of shape `(m, k)`. Entries are `inf` where fewer than `k` neighbors exist.

---

### `search.range_search`

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

### `search.build_tree`

```python
tree = search.build_tree(data, *, distance='euclidean', tree='auto', seed=None)
```

Builds a search index for repeated queries.

**Parameters**

- `data` : ndarray `(n, d)`
- `distance` : str, optional
- `tree` : `{'auto', 'vp', 'kd', 'cover'}`
  - `'auto'` chooses `'kd'` for Euclidean-like distances on low-dimensional inputs, otherwise `'vp'`.
  - `'vp'` builds a persistent VP-tree index.
  - `'cover'` builds a persistent cover-tree index.
  - `'kd'` uses the existing KD-tree-based search code path for repeated `knn` queries.
- `seed` : int or None, optional
  - RNG seed for VP-tree or cover-tree construction.

**Returns**

- `SearchIndex`

The returned object supports:

- `tree.knn(k, exclude_self=False)`
- `tree.radius_search(radius, exclude_self=False)`

For `tree='kd'`, only `knn` is supported; `radius_search` raises an error.

Both VP-tree and cover-tree are exact only for metric distances.

---

## fuel.cluster

```python
import fuel.cluster as cluster
```

### `cluster.kmeans`

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kmeans(
    data, *, k,
    variant='simp_hamerly',
    max_iter=300,
    tol=0,
    seed=None,
    init=None,
)
```

K-means clustering (Euclidean). Input must be a dense ndarray.

**Variants:** `lloyd`, `lloyd_blas`, `lloyd_naive`, `elkan`, `simp_elkan`
(`simplified_elkan`), `hamerly`, `simp_hamerly` (`simplified_hamerly`),
`exponion`, `shallot`, `hartigan_wong`, `hartigan_wong_quick`, `macqueen`.

**`init`:** `'random'`, `'first'`, `'kmeans++'`, `'kgeometric++'`, or a
`(k, d)` ndarray of fixed initial centers.

---

### `cluster.kmedians`

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kmedians(
    data, *, k, max_iter=300, tol=0, seed=None, init=None,
)
```

K-medians clustering (Manhattan distance).

---

### `cluster.kgeometric`

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kgeometric(
    data, *, k, steps, variant='default', max_iter=300, tol=1e-4, seed=None, init=None,
)
```

K-geometric-means. `steps` controls the number of geometric update sub-steps.
**Variants:** `'default'`, `'sh'` (Hamerly-accelerated).

---

### `cluster.kgmedians`

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kgmedians(
    data, *, k, gamma, alpha, max_iter=300, tol=1e-4, seed=None, init=None,
)
```

Generalised k-medians with parameters `gamma` and `alpha`.

---

### `cluster.kharmonic`

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.kharmonic(
    data, *, k, p, max_iter=300, tol=1e-4, seed=None, init=None,
)
```

K-harmonic means with harmonic power `p`.

---

### `cluster.tkmeans`

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.tkmeans(
    data, *, k, alpha, max_iter=300, tol=0, seed=None, init=None,
)
```

Trimmed k-means. `alpha` is the trimming proportion in `[0, 1)`.

---

### `cluster.fuzzycmeans`

```python
centers, membership, assignments, iterations, loss = cluster.fuzzycmeans(
    data, *, k, m, max_iter=300, tol=1e-4, seed=None, init=None,
)
```

Fuzzy c-means (Lloyd update). `m` is the fuzziness exponent (>1).
Returns membership matrix of shape `(n, k)` in addition to hard assignments.

---

### `cluster.spherical_kmeans`

```python
centers, assignments, iterations, inertia, inertia_bound = cluster.spherical_kmeans(
    data, *, k,
    variant='lloyd',
    max_iter=300,
    tol=0,
    seed=None,
    init=None,
)
```

Spherical k-means (cosine distance). Accepts dense ndarray or CSR sparse matrix.

**Variants:** `lloyd`, `elkan`, `simp_elkan`, `hamerly`, `simp_hamerly`,
`shamerly`, `selkan`.

---

### `cluster.em`

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

### `cluster.von_mises_fisher`

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

### `cluster.hierarchical`

```python
result = cluster.hierarchical(
    data,
    variant='agnes',
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
labels = result.cut_by_number_of_clusters(k)       # int64 array
labels = result.cut_by_height(height)               # int64 array
Z      = result.to_scipy_linkage()                  # (n-1, 4) float array
```

---

### `cluster.hdbscan`

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

### `cluster.dbscan`

```python
labels = cluster.dbscan(data, eps, min_points, *, distance=None, seed=None)
```

DBSCAN. Returns `int64` labels; `-1` indicates noise.

---

### `cluster.parallel_dbscan`

```python
labels = cluster.parallel_dbscan(data, eps, min_points, *, distance=None, seed=None)
```

Parallel DBSCAN. Equivalent result to `dbscan`, faster on multi-core hardware.

---

### `cluster.optics`

```python
result = cluster.optics(data, eps, min_points, *, distance=None, seed=None)
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

## fuel.outlier

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

### Angle-based

| Function | Parameters | Notes |
|----------|-----------|-------|
| `angle_based_outlier_detection(data, *, kernel, distance)` | `kernel`: `'poly2'` (default), `'poly3'`, `'linear'` | ABOD |
| `fast_angle_based_outlier_detection(data, k, *, kernel, seed, distance)` | | FastABOD |
| `locality_based_abod(data, k, l, *, distance)` | | LB-ABOD |

### Correlation / subspace

| Function | Parameters | Notes |
|----------|-----------|-------|
| `approximate_local_correlation_integral(data, nmin, alpha, g, *, seed, distance)` | | ALOCI |
| `local_correlation_integral(data, rmax, nmin, alpha, *, seed, distance)` | | LOCI |
| `correlation_outlier_probabilities(data, k, expect, dist, *, seed, distance)` | `dist`: `'chi2'` or `'gamma'` | COP |
| `local_intrinsic_dimensionality(data, k, *, estimator, seed, distance)` | | LID-based |
| `intrinsic_dimensionality_outlier_score(data, k_c, k_r, *, estimator, seed, distance)` | | IDOS |
| `subspace_outlier_degree(data, k, alpha, *, seed, distance)` | | SOD |

### Distance / density based

| Function | Parameters |
|----------|-----------|
| `k_nearest_neighbors_outlier(data, k, *, seed, distance)` | kNN distance outlier |
| `k_nearest_neighbors_distance_deviation(data, k, *, seed, distance)` | kNNDD |
| `k_nearest_neighbors_sos(data, k, *, seed, distance)` | kNN-SOS |
| `weighted_knn(data, k, *, seed, distance)` | Weighted kNN |
| `local_outlier_factor(data, k, *, seed, distance)` | LOF |
| `simplified_lof(data, k, *, seed, distance)` | Simplified LOF |
| `flexible_lof(data, krefer, kreach, *, seed, distance)` | Flexible LOF |
| `local_density_outlier_factor(data, k, *, seed, distance)` | LDOF |
| `local_outlier_probabilities(data, k, m, *, seed, distance)` | LoOP |
| `dynamic_window_outlier_factor(data, k, delta, *, seed, distance)` | DWOF |
| `local_density_factor(data, k, h, c, kernel, *, seed, distance)` | LDF |
| `simple_kernel_density_lof(data, k, h, kernel, *, seed, distance)` | KDEOS |
| `stochastic_outlier_selection(data, perplexity, *, seed, distance)` | SOS |
| `outlier_detection_independence_neighbor(data, k, *, seed, distance)` | ODIN |
| `local_isolation_coefficient(data, k, *, seed, distance)` | LIC |
| `influence_outlier(data, k, m, *, seed, distance)` | |
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

## fuel.evaluation

```python
import fuel.evaluation as evaluation
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

| Function | Returns |
|----------|---------|
| `auc(scores, labels)` | AUROC |
| `average_precision(scores, labels)` | AP |
| `auprc(scores, labels)` | Area under PR curve |
| `pr_curve(scores, labels)` | dict: recall, precision (1-D arrays) |
| `prg_auc(scores, labels)` | Area under PR-gain curve |
| `dcg(scores, labels)` | DCG |
| `ndcg(scores, labels)` | NDCG |
| `maximum_f1(scores, labels)` | Max F1 across thresholds |
| `precision_at_k(scores, labels, k)` | Precision@k |
| `r_precision(scores, labels)` | R-precision |
