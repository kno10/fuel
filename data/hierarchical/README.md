# Hierarchical Clustering Regression Sets

Each CSV exposes columns for features plus a `label` column containing the ground truth cluster indices.

- **balanced_gaussians** (268 points, 5 labels): Four moderately separated spherical clusters plus scattered noise; noise is labeled as cluster 4.
- **mixed_density_ellipses** (175 points, 3 labels): One tight cluster, one sparse elliptic cluster, and a lossy chain to test elongated merges.
- **nested_clusters** (260 points, 3 labels): Nested structure: dense center, intermediate band, and outer ring to exercise multi-level merges.
