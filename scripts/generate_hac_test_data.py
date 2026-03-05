#!/usr/bin/env python3
"""Generate hierarchical clustering regression datasets with ground truth."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "data" / "hierarchical"

FEATURE_NAMES = {"balanced_gaussians": ["x", "y"],
                 "mixed_density_ellipses": ["x", "y"],
                 "nested_clusters": ["x", "y", "z"]}


def _write_csv(path: Path, feature_names: Iterable[str], values: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow([*feature_names, "label"])
        for row, label in zip(values, labels):
            writer.writerow([*row, int(label)])


def _gaussian_cluster(center: np.ndarray, covariance: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    chol = np.linalg.cholesky(covariance)
    return center + rng.standard_normal((size, center.size)) @ chol.T


def _generate_balanced(rng: np.random.Generator) -> dict:
    centers = [np.array([0.0, 0.0]), np.array([5.0, 0.2]), np.array([0.2, 5.2]), np.array([5.0, 5.0])]
    covariance = np.array([[0.4, 0.0], [0.0, 0.4]])
    points = []
    labels = []
    for idx, center in enumerate(centers):
        cluster = _gaussian_cluster(center, covariance, 60 + (idx % 2) * 5, rng)
        points.append(cluster)
        labels.append(np.full(cluster.shape[0], idx, dtype=int))

    noise = rng.uniform(-2.5, 7.5, size=(18, 2))
    points.append(noise)
    labels.append(np.full(noise.shape[0], 4, dtype=int))

    features = np.vstack(points)
    labels = np.concatenate(labels)
    description = "Four moderately separated spherical clusters plus scattered noise; noise is labeled as cluster 4."
    return {"name": "balanced_gaussians", "values": features, "labels": labels, "description": description}


def _generate_mixed_density(rng: np.random.Generator) -> dict:
    anisotropic_cov = np.array([[1.6, 0.9], [0.9, 0.6]])
    compact_cov = np.array([[0.25, 0.0], [0.0, 0.12]])
    center_dense = np.array([-2.0, 2.5])
    center_sparse = np.array([3.5, 1.0])
    center_chain = np.array([1.5, -3.0])

    dense = _gaussian_cluster(center_dense, compact_cov, 70, rng)
    sparse = _gaussian_cluster(center_sparse, anisotropic_cov, 45, rng)

    # Bridge-style points forming an elongated chain between two centers
    chain = []
    labels = []
    for idx, base in enumerate(np.linspace(center_chain, center_sparse, 5)):
        segment = _gaussian_cluster(base, np.array([[0.4, 0.0], [0.0, 0.4]]), 12, rng)
        chain.append(segment)
        labels.append(np.full(segment.shape[0], 2, dtype=int))

    features = np.vstack([dense, sparse, np.vstack(chain)])
    labels = np.concatenate([np.full(dense.shape[0], 0, dtype=int),
                             np.full(sparse.shape[0], 1, dtype=int),
                             np.concatenate(labels)])
    description = "One tight cluster, one sparse elliptic cluster, and a lossy chain to test elongated merges."
    return {"name": "mixed_density_ellipses", "values": features, "labels": labels, "description": description}


def _generate_nested(rng: np.random.Generator) -> dict:
    inner_xy = rng.normal(scale=0.35, size=(80, 2))
    inner = np.column_stack(
        [inner_xy[:, 0], inner_xy[:, 1], rng.normal(scale=0.1, size=80) - 2.5]
    )

    band_angles = rng.uniform(0, 2 * np.pi, size=60)
    band_radius = rng.uniform(2.4, 3.0, size=60)
    band = np.column_stack(
        [
            band_radius * np.cos(band_angles),
            band_radius * np.sin(band_angles),
            rng.normal(scale=0.1, size=60),
        ]
    )

    radius = rng.uniform(5.5, 6.5, size=120)
    angles = rng.uniform(0, 2 * np.pi, size=120)
    ring = np.column_stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            rng.normal(scale=0.1, size=120) + 2.5,
        ]
    )

    features = np.vstack([inner, band, ring])
    labels = np.concatenate([
        np.zeros(inner.shape[0], dtype=int),
        np.full(band.shape[0], 1, dtype=int),
        np.full(ring.shape[0], 2, dtype=int),
    ])
    description = "Nested structure: dense center, intermediate band, and outer ring to exercise multi-level merges."
    return {"name": "nested_clusters", "values": features, "labels": labels, "description": description}


def main() -> None:
    rng = np.random.default_rng(2026)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generators = [_generate_balanced, _generate_mixed_density, _generate_nested]
    summary = []

    for gen in generators:
        entry = gen(rng)
        name = entry["name"]
        path = OUTPUT_DIR / f"{name}.csv"
        feature_names = FEATURE_NAMES[name]
        _write_csv(path, feature_names, entry["values"], entry["labels"])
        summary.append({"name": name, "path": path, "description": entry["description"],
                        "points": entry["values"].shape[0], "clusters": len(np.unique(entry["labels"]))})

    readme_path = OUTPUT_DIR / "README.md"
    with readme_path.open("w") as readme:
        readme.write("# Hierarchical Clustering Regression Sets\n\n")
        readme.write("Each CSV exposes columns for features plus a `label` column containing the ground truth cluster indices.\n\n")
        for line in summary:
            readme.write(f"- **{line['name']}** ({line['points']} points, {line['clusters']} labels): {line['description']}\n")

    print("Generated hierarchical clustering datasets:")
    for dataset in summary:
        print(f"  - {dataset['name']} -> {dataset['path']} ({dataset['points']} points, {dataset['clusters']} clusters)")


if __name__ == "__main__":
    main()
