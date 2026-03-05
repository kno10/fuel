#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.cluster import hierarchy

from file_loader import load_numeric_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AGNES benchmark using SciPy for comparison"
    )
    parser.add_argument("data_path", type=Path, help="Path to CSV or whitespace-separated file")
    parser.add_argument("k", type=int, help="Number of clusters to form from the hierarchy")
    parser.add_argument(
        "linkage",
        type=str,
        choices=[
            "single",
            "complete",
            "average",
            "ward",
            "centroid",
            "median",
            "weighted",
        ],
        help="Linkage method to use",
    )
    return parser.parse_args()


def summarize_cluster_sizes(labels: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    cluster_ids, counts = np.unique(labels, return_counts=True)

    cluster_sizes: list[tuple[int, int]] = []
    for cluster_id, count in zip(cluster_ids.tolist(), counts.tolist()):
        cluster_sizes.append((int(cluster_id), int(count)))

    cluster_sizes.sort(key=lambda item: item[0])
    return cluster_sizes, 0  # hierarchical clustering has no noise label


def format_cluster_sizes(cluster_sizes: list[tuple[int, int]]) -> str:
    if not cluster_sizes:
        return "none"
    return ",".join(f"{cluster_id}:{size}" for cluster_id, size in cluster_sizes)


def main() -> int:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("k must be greater than 0")

    data = load_numeric_data(args.data_path)
    n = data.shape[0]
    if n < 2:
        raise ValueError("CSV must contain at least two rows")

    start = time.perf_counter()
    Z = hierarchy.linkage(data, method=args.linkage, metric="euclidean")
    # cut the dendrogram to get exactly k clusters
    labels = hierarchy.fcluster(Z, t=args.k, criterion="maxclust")
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    cluster_sizes, noise_count = summarize_cluster_sizes(labels)

    print(f"time_ms={elapsed_ms:.3f}")
    print(f"cluster_count={len(cluster_sizes)}")
    print(f"noise_count={noise_count}")
    print(f"cluster_sizes={format_cluster_sizes(cluster_sizes)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
