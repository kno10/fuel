#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from file_loader import load_numeric_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="kNN outlier benchmark using scikit-learn for comparison"
    )
    parser.add_argument("data_path", type=Path, help="Path to CSV or whitespace-separated file")
    parser.add_argument("k", type=int, help="Neighbor rank for outlier score")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.k <= 0:
        raise ValueError("k must be greater than 0")

    data = load_numeric_data(args.data_path)
    n = data.shape[0]
    if n < 2:
        raise ValueError("CSV must contain at least two rows")

    k_effective = min(args.k, n - 1)
    n_neighbors = min(args.k + 1, n)

    start = time.perf_counter()
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(data)
    distances, _ = nn.kneighbors(data, return_distance=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    avg_score = np.mean(distances[:, k_effective])

    print(f"time_ms={elapsed_ms:.3f}")
    print(f"avg_score={avg_score:.12f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
