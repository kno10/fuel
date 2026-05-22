import argparse
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fuel import get_rayon_parallelism, search

def run_fuel_search(data, queries, tree_name, k):
    start_build = time.perf_counter()
    index = search.SearchIndex(data, distance='euclidean', tree=tree_name, seed=42)
    build_time = time.perf_counter() - start_build
    start_query = time.perf_counter()
    _, distances = index.knn(queries, k, exclude_self=False)
    query_time = time.perf_counter() - start_query
    return build_time, query_time, distances

def run_sklearn_tree(algorithm, data, queries, k, **kwargs):
    start_build = time.perf_counter()
    tree = NearestNeighbors(n_neighbors=k, algorithm=algorithm, metric='euclidean', **kwargs)
    tree.fit(data)
    build_time = time.perf_counter() - start_build
    start_query = time.perf_counter()
    distances, _ = tree.kneighbors(queries, n_neighbors=k)
    query_time = time.perf_counter() - start_query
    return build_time, query_time, distances

def main():
    parser = argparse.ArgumentParser(description='Compare Fuel search trees against sklearn trees.')
    parser.add_argument('csv', nargs='?', help='Path to CSV file to load as data.')
    parser.add_argument('k', type=int, help='Number of nearest neighbors to query.')
    parser.add_argument('--query-csv', help='Path to CSV file to load as query data.')
    parser.add_argument('--skip-header', action='store_true', help='Skip the first row of the data CSV file.')
    parser.add_argument('--n', type=int, default=5000, help='Number of random samples to generate when no CSV is provided.')
    parser.add_argument('--d', type=int, default=6, help='Number of dimensions for random data when no CSV is provided.')
    args = parser.parse_args()

    if args.csv:
        data = np.loadtxt(args.csv, delimiter=',', skiprows=1 if args.skip_header else 0)
        data = data.astype(np.float64)
        print(f"Loaded CSV data from {args.csv}: shape={data.shape}")
    else:
        rng = np.random.RandomState(0)
        data = rng.normal(size=(args.n, args.d)).astype(np.float64)
        print(f"Using random dataset: n={args.n}, d={args.d}")

    if args.query_csv:
        query_data = np.loadtxt(args.query_csv, delimiter=',', skiprows=1 if args.skip_header else 0)
        query_data = query_data.astype(np.float64)
        print(f"Loaded query CSV data from {args.query_csv}: shape={query_data.shape}")
    else:
        query_data = data

    print(f"Using Euclidean distance and k={args.k}")

    for tree_name in ['vp', 'kd', 'cover']:
        build_time, query_time, distances = run_fuel_search(data, query_data, tree_name, args.k)
        total_time = build_time + query_time
        avg_dist = float(np.asarray(distances).mean())
        print(
            f"fuel {tree_name:8} | build: {build_time:.4f}s | query: {query_time:.4f}s | "
            f"total: {total_time:.4f}s | avg kNN dist: {avg_dist:.8f}"
        )

    build_time, query_time, distances = run_fuel_search(data, query_data, 'linear', args.k)
    total_time = build_time + query_time
    avg_dist = float(np.asarray(distances).mean())
    print(
        f"fuel-linear   | build: {build_time:.4f}s | query: {query_time:.4f}s | "
        f"total: {total_time:.4f}s | avg kNN dist: {avg_dist:.8f}"
    )

    sklearn_configs = [
        ('sklearn-brute', 'brute'),
        ('sklearn-kd', 'kd_tree'),
        ('sklearn-ball', 'ball_tree'),
    ]
    for label, algorithm in sklearn_configs:
        build_time, query_time, distances = run_sklearn_tree(algorithm, data, query_data, args.k)
        total_time = build_time + query_time
        avg_dist = float(np.asarray(distances).mean())
        print(
            f"{label:13} | build: {build_time:.4f}s | query: {query_time:.4f}s | "
            f"total: {total_time:.4f}s | avg kNN dist: {avg_dist:.8f}"
        )

if __name__ == '__main__':
    main()
