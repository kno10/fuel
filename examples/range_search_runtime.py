import argparse
import time
import numpy as np
from sklearn.neighbors import BallTree, KDTree
from fuel import search

def run_fuel_tree(data, queries, tree_name, radius):
    start_build = time.perf_counter()
    index = search.build_tree(data, distance='euclidean', tree=tree_name, seed=42)
    build_time = time.perf_counter() - start_build
    start_query = time.perf_counter()
    _, distances = index.radius_search(queries, radius, exclude_self=False)
    query_time = time.perf_counter() - start_query
    return build_time, query_time, distances

def run_sklearn_tree(tree_cls, data, queries, radius, **kwargs):
    start_build = time.perf_counter()
    tree = tree_cls(data, metric='euclidean', **kwargs)
    build_time = time.perf_counter() - start_build
    start_query = time.perf_counter()
    _, distances = tree.query_radius(queries, r=radius, return_distance=True)
    query_time = time.perf_counter() - start_query
    return build_time, query_time, distances

def summarize_range_results(distances):
    counts = np.array([len(row) for row in distances], dtype=np.int64)
    all_distances = np.concatenate(distances) if len(distances) > 0 else np.array([], dtype=np.float64)
    avg_count = float(counts.mean()) if counts.size > 0 else 0.0
    avg_dist = float(all_distances.mean()) if all_distances.size > 0 else float('nan')
    return avg_count, avg_dist

def main():
    parser = argparse.ArgumentParser(description='Compare Fuel radius search against sklearn radius search.')
    parser.add_argument('csv', nargs='?', help='Path to CSV file to load as data.')
    parser.add_argument('radius', nargs='?', type=float, help='Search radius for range queries.')
    parser.add_argument('--query-csv', help='Path to CSV file to load as query data.')
    parser.add_argument('--skip-header', action='store_true', help='Skip the first row of the data CSV file.')
    parser.add_argument('--n', type=int, default=5000, help='Number of random samples to generate when no CSV is provided.')
    parser.add_argument('--d', type=int, default=6, help='Number of dimensions for random data when no CSV is provided.')
    args = parser.parse_args()

    if args.radius is None:
        if args.csv is None:
            parser.error('the following arguments are required: radius')
        try:
            args.radius = float(args.csv)
            args.csv = None
        except ValueError:
            parser.error('the following arguments are required: radius')

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

    print(f"Using Euclidean radius search with radius={args.radius}")

    for tree_name in ['vp', 'kd', 'cover']:
        build_time, query_time, distances = run_fuel_tree(data, query_data, tree_name, args.radius)
        avg_count, avg_dist = summarize_range_results(distances)
        total_time = build_time + query_time
        print(
            f"fuel {tree_name:7} | build: {build_time:.4f}s | query: {query_time:.4f}s | "
            f"total: {total_time:.4f}s | avg count: {avg_count:.4f} | avg dist: {avg_dist:.8f}"
        )

    sklearn_configs = [
        ('sklearn-kd', KDTree, {}),
        ('sklearn-ball', BallTree, {}),
    ]
    for label, tree_cls, kwargs in sklearn_configs:
        build_time, query_time, distances = run_sklearn_tree(tree_cls, data, query_data, args.radius, **kwargs)
        avg_count, avg_dist = summarize_range_results(distances)
        total_time = build_time + query_time
        print(
            f"{label:12} | build: {build_time:.4f}s | query: {query_time:.4f}s | "
            f"total: {total_time:.4f}s | avg count: {avg_count:.4f} | avg dist: {avg_dist:.8f}"
        )

if __name__ == '__main__':
    main()
