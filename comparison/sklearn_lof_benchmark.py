#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from file_loader import load_numeric_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOF outlier benchmark using scikit-learn for comparison"
    )
    parser.add_argument("data_path", type=Path, help="Path to CSV or whitespace-separated file")
    parser.add_argument("k", type=int, help="Neighbor count for LOF")
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

    start = time.perf_counter()
    lof = LocalOutlierFactor(n_neighbors=k_effective, metric="euclidean")
    lof.fit_predict(data)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    avg_score = np.mean(-lof.negative_outlier_factor_)

    print(f"time_ms={elapsed_ms:.3f}")
    print(f"avg_score={avg_score:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
