"""
K-medoids clustering examples.

Demonstrates:
- kmedoids with various algorithm variants
- Coordinate data vs precomputed distance matrix
- dynmsc / silhouette_clustering for automatic k selection
- kmedians and kgeometric medians
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from fuel.cluster import (
    kmedoids,
    dynmsc,
    silhouette_clustering,
    kmedians,
    kgeometric,
)


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    blobs = [
        rng.normal([0, 0], 0.5, (50, 2)),
        rng.normal([5, 0], 0.5, (50, 2)),
        rng.normal([2.5, 4], 0.5, (50, 2)),
    ]
    return np.vstack(blobs).astype(np.float64)


data = make_data()
n = len(data)

# ---------------------------------------------------------------------------
# 1. Basic kmedoids with coordinate data
# ---------------------------------------------------------------------------
print("=== kmedoids - coordinate data ===")
for variant in ['fasterpam', 'fastpam1', 'pam', 'par_fasterpam']:
    result = kmedoids(data, medoids=3, variant=variant, seed=42)
    k = result.labels.max() + 1
    print(f"  {variant:16} loss={result.loss:.3f}  k={k}  "
          f"iters={result.n_iter}  swaps={result.n_swap}")
    assert k == 3, f"{variant}: expected k=3, got {k}"

# ---------------------------------------------------------------------------
# 2. kmedoids with precomputed square distance matrix
# ---------------------------------------------------------------------------
print("\n=== kmedoids - precomputed square matrix ===")
D_square = squareform(pdist(data, metric='euclidean'))
result_sq = kmedoids(D_square, medoids=3, distance='precomputed', seed=42)
k = result_sq.labels.max() + 1
print(f"  square matrix: k={k}, loss={result_sq.loss:.3f}")
assert k == 3

# ---------------------------------------------------------------------------
# 3. kmedoids with precomputed condensed vector
# ---------------------------------------------------------------------------
print("\n=== kmedoids - precomputed condensed vector ===")
D_cond = pdist(data, metric='euclidean')
result_cd = kmedoids(D_cond, medoids=3, distance='precomputed', seed=42)
k = result_cd.labels.max() + 1
print(f"  condensed vector: k={k}, loss={result_cd.loss:.3f}")
assert k == 3

# Square and condensed should produce the same loss
assert abs(result_sq.loss - result_cd.loss) < 1e-6, (
    f"square vs condensed loss mismatch: {result_sq.loss} != {result_cd.loss}"
)
print("  square == condensed: ok")

# ---------------------------------------------------------------------------
# 4. kmedoids with PAM BUILD initialization
# ---------------------------------------------------------------------------
print("\n=== kmedoids - PAM BUILD init ===")
result_build = kmedoids(data, medoids=3, variant='fasterpam', init='build', seed=42)
print(f"  BUILD init: loss={result_build.loss:.3f}, iters={result_build.n_iter}")

# ---------------------------------------------------------------------------
# 5. dynmsc - automatic k selection
# ---------------------------------------------------------------------------
print("\n=== dynmsc - automatic k ===")
result = dynmsc(data, 8, minimum_k=2, seed=42)
print(f"  best k={result.bestk}, loss={result.loss:.3f}")
print(f"  medoid silhouette over k={list(result.rangek)}: "
      f"{[f'{v:.3f}' for v in result.losses]}")
# On this well-separated dataset, dynmsc should find k=3
assert result.bestk == 3, f"dynmsc expected bestk=3, got {result.bestk}"

# ---------------------------------------------------------------------------
# 6. medoid silhouette_clustering - explicit k
# ---------------------------------------------------------------------------
print("\n=== silhouette_clustering ===")
result = silhouette_clustering(data, 3, seed=42)
print(f"  best k={result.medoids.shape[0]}, loss={result.loss:.3f}")

# ---------------------------------------------------------------------------
# 7. kmedians (L1 centers)
# ---------------------------------------------------------------------------
print("\n=== kmedians ===")
result = kmedians(data, k=3, seed=42)
k = result.labels.max() + 1
print(f"  k={k}, inertia={result.inertia:.3f}, iters={result.n_iter}")
assert k == 3

# ---------------------------------------------------------------------------
# 8. kgeometric (geometric median centers)
# ---------------------------------------------------------------------------
print("\n=== kgeometric (geometric median) ===")
result = kgeometric(data, k=3, steps=5, seed=42)
k = result.labels.max() + 1
print(f"  k={k}, inertia={result.inertia:.3f}, iters={result.n_iter}")
assert k == 3

# ---------------------------------------------------------------------------
# 9. Result attributes
# ---------------------------------------------------------------------------
print("\n=== result attributes ===")
result = kmedoids(data, medoids=3, seed=42)
print(f"  type:    {type(result).__name__}")
print(f"  loss:    {result.loss:.4f}")
print(f"  medoids: {sorted(result.medoids.tolist())}")
print(f"  labels shape: {result.labels.shape}")
assert result.labels.shape == (n,)
assert result.medoids.shape == (3,)

print("\nAll k-medoids examples completed successfully.")
