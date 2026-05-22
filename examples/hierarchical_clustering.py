"""
Hierarchical clustering examples.

Demonstrates:
- Coordinate-based clustering with various variants and linkages
- Precomputed distance input (square matrix and condensed vector)
- Cutting the dendrogram by cluster count and by height
- Search-tree-accelerated single-link variants
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from fuel.cluster import hierarchical


def make_data(seed=0):
    rng = np.random.default_rng(seed)
    blobs = [
        rng.normal([0, 0], 0.4, (40, 2)),
        rng.normal([3, 0], 0.4, (40, 2)),
        rng.normal([1.5, 2.5], 0.4, (40, 2)),
    ]
    return np.vstack(blobs).astype(np.float64)


def check_labels(labels, expected_k=3):
    k = labels.max() + 1
    assert k == expected_k, f"expected {expected_k} clusters, got {k}"
    return k


# ---------------------------------------------------------------------------
# 1. Standard variants with Ward linkage
# ---------------------------------------------------------------------------
data = make_data()
print("=== Coordinate-based (Ward) ===")
for variant in ['agnes', 'anderberg', 'muellner', 'nn_chain']:
    h = hierarchical(data, variant=variant, linkage='ward')
    labels = h.cut_by_number_of_clusters(3)
    k = check_labels(labels)
    print(f"  {variant:12} -> {k} clusters")

# ---------------------------------------------------------------------------
# 2. Various linkages with nn_chain
# ---------------------------------------------------------------------------
print("\n=== nn_chain, various linkages ===")
for linkage in ['single', 'complete', 'average', 'ward', 'median']:
    h = hierarchical(data, variant='nn_chain', linkage=linkage)
    labels = h.cut_by_number_of_clusters(3)
    k = labels.max() + 1
    print(f"  {linkage:12} -> {k} clusters")

# ---------------------------------------------------------------------------
# 3. Set-based variants (extended linkage set)
# ---------------------------------------------------------------------------
print("\n=== set-based variants (minimax) ===")
for variant in ['set_agnes', 'set_muellner', 'set_nn_chain']:
    h = hierarchical(data, variant=variant, linkage='minimax')
    labels = h.cut_by_number_of_clusters(3)
    k = labels.max() + 1
    print(f"  {variant:14} -> {k} clusters")

# ---------------------------------------------------------------------------
# 4. Cut by height
# ---------------------------------------------------------------------------
print("\n=== cut_by_height ===")
h = hierarchical(data, variant='nn_chain', linkage='ward')
# Pick a height that yields a small number of clusters
labels = h.cut_by_height(2.0)
print(f"  height=2.0 -> {labels.max() + 1} clusters")

# ---------------------------------------------------------------------------
# 5. SLINK and CLINK (linear-memory)
# ---------------------------------------------------------------------------
print("\n=== slink / clink ===")
h_slink = hierarchical(data, variant='slink')
h_clink = hierarchical(data, variant='clink')
labels_s = h_slink.cut_by_number_of_clusters(3)
labels_c = h_clink.cut_by_number_of_clusters(3)
print(f"  slink -> {labels_s.max() + 1} clusters")
print(f"  clink -> {labels_c.max() + 1} clusters")

# ---------------------------------------------------------------------------
# 6. Precomputed square distance matrix
# ---------------------------------------------------------------------------
print("\n=== precomputed square matrix ===")
D_square = squareform(pdist(data, metric='euclidean'))
for variant in ['agnes', 'nn_chain', 'set_agnes']:
    h = hierarchical(D_square, variant=variant, linkage='ward',
                     distance='precomputed')
    labels = h.cut_by_number_of_clusters(3)
    k = check_labels(labels)
    print(f"  {variant:12} -> {k} clusters")

# SLINK on precomputed square matrix
h = hierarchical(D_square, variant='slink', distance='precomputed')
labels = h.cut_by_number_of_clusters(3)
print(f"  {'slink':12} -> {labels.max() + 1} clusters")

# ---------------------------------------------------------------------------
# 7. Precomputed condensed distance vector
# ---------------------------------------------------------------------------
print("\n=== precomputed condensed vector ===")
D_cond = pdist(data, metric='euclidean')  # length n*(n-1)/2
for variant in ['agnes', 'nn_chain', 'set_agnes', 'slink']:
    h = hierarchical(D_cond, variant=variant, linkage='ward',
                     distance='precomputed')
    labels = h.cut_by_number_of_clusters(3)
    k = labels.max() + 1
    print(f"  {variant:12} -> {k} clusters")

# ---------------------------------------------------------------------------
# 8. Consistency: square vs condensed should give identical results
# ---------------------------------------------------------------------------
print("\n=== square vs condensed consistency ===")
h_sq = hierarchical(D_square, variant='agnes', linkage='ward', distance='precomputed')
h_cd = hierarchical(D_cond,   variant='agnes', linkage='ward', distance='precomputed')
labels_sq = h_sq.cut_by_number_of_clusters(3)
labels_cd = h_cd.cut_by_number_of_clusters(3)
# Cluster assignments may differ in label index; compare via sorted sizes
sizes_sq = sorted(np.bincount(labels_sq).tolist())
sizes_cd = sorted(np.bincount(labels_cd).tolist())
assert sizes_sq == sizes_cd, f"square vs condensed mismatch: {sizes_sq} != {sizes_cd}"
print(f"  ok - cluster sizes: {sizes_sq}")

# ---------------------------------------------------------------------------
# 9. Search-tree single-link
# ---------------------------------------------------------------------------
print("\n=== search-tree single-link ===")
for variant in ['heap_of_searchers_single_link', 'boruvka_searchers_single_link']:
    h = hierarchical(data, variant=variant, distance='euclidean')
    labels = h.cut_by_number_of_clusters(3)
    k = labels.max() + 1
    print(f"  {variant} -> {k} clusters")

print("\nAll hierarchical clustering examples completed successfully.")
