"""
Outlier detection examples.

Demonstrates:
- LOF, simplified LOF, ABOD, FastABOD
- kNN-outlier, distance-from-center
- Isolation Forest
- Using a prebuilt SearchIndex for multiple algorithms
- Verifying that injected outliers receive high scores
"""

import numpy as np

from fuel import search
from fuel.outlier import (
    local_outlier_factor,
    simplified_lof,
    angle_based_outlier_detection,
    fast_angle_based_outlier_detection,
    k_nearest_neighbors_outlier,
    distance_from_center,
    isolation_forest,
    local_outlier_probabilities,
)


def make_data(n_inliers=200, n_outliers=10, seed=0):
    """Gaussian inliers with a few far-away outliers."""
    rng = np.random.default_rng(seed)
    inliers = rng.normal(0.0, 1.0, (n_inliers, 4)).astype(np.float32)
    outliers = rng.uniform(6.0, 10.0, (n_outliers, 4)).astype(np.float32)
    data = np.vstack([inliers, outliers])
    true_outlier_idx = set(range(n_inliers, n_inliers + n_outliers))
    return data, true_outlier_idx


def top_k_recall(scores, true_outlier_idx, k):
    """Fraction of injected outliers found in the top-k scored points."""
    top_k = set(np.argsort(scores)[-k:].tolist())
    return len(top_k & true_outlier_idx) / len(true_outlier_idx)


data, true_idx = make_data()
n_outliers = len(true_idx)
k = 10

print(f"Dataset: {data.shape[0]} points ({data.shape[0] - n_outliers} inliers, "
      f"{n_outliers} injected outliers), d={data.shape[1]}")

# ---------------------------------------------------------------------------
# Build a shared search index for algorithms that need one
# ---------------------------------------------------------------------------
index = search.SearchIndex(data, distance='euclidean', precompute=k + 1)

# ---------------------------------------------------------------------------
# LOF
# ---------------------------------------------------------------------------
result = local_outlier_factor(data, k=k, index=index)
recall = top_k_recall(result.scores, true_idx, k=n_outliers)
print(f"\nLOF            recall@{n_outliers}={recall:.2f}  "
      f"(ascending={result.metadata['ascending']})")
assert recall >= 0.8, f"LOF recall too low: {recall}"

# ---------------------------------------------------------------------------
# Simplified LOF
# ---------------------------------------------------------------------------
result = simplified_lof(data, k=k, index=index)
recall = top_k_recall(result.scores, true_idx, k=n_outliers)
print(f"SimplifiedLOF  recall@{n_outliers}={recall:.2f}")

# ---------------------------------------------------------------------------
# kNN-outlier (kth-nearest-neighbor distance)
# ---------------------------------------------------------------------------
result = k_nearest_neighbors_outlier(data, k=k, index=index)
recall = top_k_recall(result.scores, true_idx, k=n_outliers)
print(f"kNN-outlier    recall@{n_outliers}={recall:.2f}")
assert recall >= 0.8, f"kNN-outlier recall too low: {recall}"

# ---------------------------------------------------------------------------
# ABOD (angle-based; works without a tree)
# ---------------------------------------------------------------------------
# Use a smaller subset to keep ABOD tractable (O(n^3))
data_small = data[:80]
true_idx_small = {i for i in true_idx if i < 80}
result = angle_based_outlier_detection(data_small)
ascending = result.metadata['ascending']
if len(true_idx_small) > 0:
    # ABOD: lower variance = more outlying, so ascending=False
    recall = top_k_recall(result.scores, true_idx_small, k=len(true_idx_small))
    print(f"\nABOD (n=80)    recall@{len(true_idx_small)}={recall:.2f}  "
          f"(ascending={ascending})")

# ---------------------------------------------------------------------------
# FastABOD
# ---------------------------------------------------------------------------
result = fast_angle_based_outlier_detection(data, k=k, index=index)
recall = top_k_recall(result.scores, true_idx, k=n_outliers)
print(f"FastABOD       recall@{n_outliers}={recall:.2f}")

# ---------------------------------------------------------------------------
# Distance from center (simple baseline)
# ---------------------------------------------------------------------------
result = distance_from_center(data)
recall = top_k_recall(result.scores, true_idx, k=n_outliers)
print(f"DistFromCenter recall@{n_outliers}={recall:.2f}")
assert recall >= 0.8, f"distance_from_center recall too low: {recall}"

# ---------------------------------------------------------------------------
# LoOP (Local Outlier Probabilities)
# ---------------------------------------------------------------------------
result = local_outlier_probabilities(data, k=k, m=5, index=index)
recall = top_k_recall(result.scores, true_idx, k=n_outliers)
print(f"LoOP           recall@{n_outliers}={recall:.2f}")

# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
result = isolation_forest(data, num_trees=100, subsample_size=64, seed=42)
recall = top_k_recall(result.scores, true_idx, k=n_outliers)
print(f"\nIsolation Forest recall@{n_outliers}={recall:.2f}  "
      f"(ascending={result.metadata['ascending']})")

# ---------------------------------------------------------------------------
# Metadata sanity checks
# ---------------------------------------------------------------------------
result = local_outlier_factor(data, k=k)
assert 'label' in result.metadata
assert 'ascending' in result.metadata
assert isinstance(result.scores, np.ndarray)
assert result.scores.shape == (data.shape[0],)
print("\nMetadata keys:", sorted(result.metadata.keys()))

print("\nAll outlier detection examples completed successfully.")
