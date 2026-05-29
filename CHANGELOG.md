# Changelog

## fuel 0.1.1 (unreleased)

- New outlier algorithms: `outlier::{cblof, dbscan_outlier, em_outlier, glosh, kmeans_outlier, silhouette_outlier, tkmeans_outlier}` (including `tkmeans_star` variant)
- Python bindings for all of the above, plus DBSCAN `noise_as_outliers`
- `em_outlier` Python binding exposes `EmConfig::prior` (MAP regularization) and `EmConfig::noise_ratio`
- `HdbscanHierarchy` Python class exposes `glosh(min_cluster_size)`
- Redesigned `cluster::parallel_dbscan`

## fuel 0.1.0 (2026-05-22)

- Happy Birthday, fuel!
- the initial release

