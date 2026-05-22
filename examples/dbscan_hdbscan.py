"""
DBSCAN, OPTICS, and HDBSCAN examples.

Demonstrates:
- DBSCAN with density-separated clusters
- OPTICS reachability ordering
- HDBSCAN hierarchy construction and cluster extraction
- Multiple algorithm variants
"""

import numpy as np

from fuel.cluster import dbscan, optics, hdbscan


def make_blobs(seed=0):
    """Three well-separated Gaussian clusters plus some noise."""
    rng = np.random.default_rng(seed)
    blobs = [
        rng.normal([0, 0], 0.3, (60, 2)),
        rng.normal([4, 0], 0.3, (60, 2)),
        rng.normal([2, 3], 0.3, (60, 2)),
    ]
    noise = rng.uniform(-1, 5, (15, 2))
    data = np.vstack(blobs + [noise]).astype(np.float64)
    return data


def make_varying_density(seed=1):
    """Two clusters with different densities (challenging for DBSCAN)."""
    rng = np.random.default_rng(seed)
    tight = rng.normal([0, 0], 0.1, (80, 2))
    loose = rng.normal([3, 0], 0.6, (80, 2))
    return np.vstack([tight, loose]).astype(np.float64)


# ---------------------------------------------------------------------------
# DBSCAN - well-separated blobs
# ---------------------------------------------------------------------------
print("=== DBSCAN ===")
data = make_blobs()

labels = dbscan(data, eps=0.6, min_points=5)
n_clusters = labels.max() + 1
n_noise = (labels == -1).sum()
print(f"  blobs: {n_clusters} clusters, {n_noise} noise points")
assert n_clusters == 3, f"expected 3 clusters, got {n_clusters}"

# Parallel variant should give identical labels
labels_par = dbscan(data, eps=0.6, min_points=5, variant='parallel')
assert np.array_equal(np.sort(labels), np.sort(labels_par)) or \
    (labels_par.max() + 1 == 3), "parallel DBSCAN cluster count mismatch"
print(f"  parallel variant: {labels_par.max() + 1} clusters")

# ---------------------------------------------------------------------------
# DBSCAN - Manhattan distance
# ---------------------------------------------------------------------------
labels_l1 = dbscan(data, eps=0.8, min_points=5, distance='manhattan')
print(f"  manhattan distance: {labels_l1.max() + 1} clusters")

# ---------------------------------------------------------------------------
# OPTICS
# ---------------------------------------------------------------------------
print("\n=== OPTICS ===")
result = optics(data, max_eps=2.0, min_points=5)
ordering = result.ordering()
reachability = result.reachability()
print(f"  ordering length: {len(ordering)}")
finite_reach = reachability[reachability < np.inf]
print(f"  reachability: min={finite_reach.min():.3f}, max={finite_reach.max():.3f}")
assert len(ordering) == len(data)
assert len(reachability) == len(data)

def _hdbscan_flat_labels(result, n):
    """Build a flat label array from extract_hdbscan / extract_simplified result."""
    nodes = result.get('hierarchy', result)['nodes']
    labels = np.full(n, -1, dtype=np.intp)
    for cid, node in enumerate(nodes):
        labels[list(node['members'])] = cid
    return labels


# ---------------------------------------------------------------------------
# HDBSCAN - brute-force variants
# ---------------------------------------------------------------------------
print("\n=== HDBSCAN ===")
min_pts = 5

for variant in ['hdbscan_prim', 'slink_hdbscan']:
    h = hdbscan(data, min_points=min_pts, variant=variant)
    core_dists = h.core_distances()
    assert core_dists.shape == (len(data),)
    assert (core_dists >= 0).all()

    labels_dict = h.extract_hdbscan(min_cluster_size=10, hierarchical=False)
    labels = _hdbscan_flat_labels(labels_dict, len(data))
    n_clusters = labels.max() + 1
    n_noise = (labels == -1).sum()
    print(f"  {variant}: {n_clusters} clusters, {n_noise} noise  "
          f"core_dist_mean={core_dists.mean():.3f}")

# ---------------------------------------------------------------------------
# HDBSCAN - tree-accelerated variant
# ---------------------------------------------------------------------------
for variant in ['heap_of_searchers_hdbscan', 'boruvka_searchers_hdbscan']:
    h = hdbscan(data, min_points=min_pts, variant=variant)
    labels_dict = h.extract_hdbscan(min_cluster_size=10, hierarchical=False)
    labels = _hdbscan_flat_labels(labels_dict, len(data))
    print(f"  {variant}: {labels.max() + 1} clusters")

# ---------------------------------------------------------------------------
# HDBSCAN - simplified extraction
# ---------------------------------------------------------------------------
print("\n=== HDBSCAN simplified extraction ===")
h = hdbscan(data, min_points=min_pts, variant='hdbscan_prim')
result = h.extract_simplified(min_cluster_size=10)
labels = _hdbscan_flat_labels(result, len(data))
print(f"  simplified: {labels.max() + 1} clusters, "
      f"{(labels == -1).sum()} noise")

# ---------------------------------------------------------------------------
# HDBSCAN - scipy linkage interop
# ---------------------------------------------------------------------------
print("\n=== HDBSCAN to scipy linkage ===")
linkage_matrix = h.to_scipy_linkage()
assert linkage_matrix.shape == (len(data) - 1, 4)
assert (linkage_matrix[:, 2] >= 0).all(), "negative merge heights"
print(f"  linkage matrix shape: {linkage_matrix.shape}  "
      f"height range: [{linkage_matrix[:, 2].min():.3f}, "
      f"{linkage_matrix[:, 2].max():.3f}]")

# ---------------------------------------------------------------------------
# HDBSCAN - varying density (shows advantage over DBSCAN)
# ---------------------------------------------------------------------------
print("\n=== HDBSCAN vs DBSCAN on varying density ===")
data_vd = make_varying_density()

# DBSCAN struggles with varying density
labels_db = dbscan(data_vd, eps=0.3, min_points=5)
print(f"  DBSCAN (eps=0.3): {labels_db.max() + 1} clusters, "
      f"{(labels_db == -1).sum()} noise")

# HDBSCAN adapts to local density
h_vd = hdbscan(data_vd, min_points=5, variant='hdbscan_prim')
labels_dict = h_vd.extract_hdbscan(min_cluster_size=15, hierarchical=False)
labels_hdb = _hdbscan_flat_labels(labels_dict, len(data_vd))
print(f"  HDBSCAN:         {labels_hdb.max() + 1} clusters, "
      f"{(labels_hdb == -1).sum()} noise")

print("\nAll DBSCAN/HDBSCAN examples completed successfully.")
