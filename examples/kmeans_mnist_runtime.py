from pathlib import Path
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

from fuel.cluster import kmeans


def load_mnist(n_samples=70000):
    print("Loading MNIST from openml...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32).copy(order="C")
    if n_samples is not None:
        X = X[:n_samples]
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    return X


def benchmark_sklearn(X, init_centers, algorithm="lloyd", n_clusters=10):
    print("Running sklearn KMeans...")
    model = KMeans(
        n_clusters=n_clusters,
        init=init_centers,
        n_init=1,
        max_iter=300,
        algorithm=algorithm,
        tol=0,
        random_state=42,
        verbose=0,
    )
    start = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - start
    import sklearn
    n_threads = sklearn.utils._openmp_helpers._openmp_effective_n_threads()
    print(f"sklearn {algorithm} finished in {elapsed:.3f} seconds n_threads={n_threads}")
    print(f"sklearn iterations={model.n_iter_} inertia={model.inertia_}")
    return elapsed


def benchmark_fuel(X, init_centers, variant="simp_hamerly", n_clusters=10):
    print("Running fuel kmeans", variant)
    start = time.perf_counter()
    result = kmeans(
        X,
        k=n_clusters,
        variant=variant,
        max_iter=300,
        tol=0,
        seed=42,
        init=init_centers,
    )
    elapsed = time.perf_counter() - start
    import fuel
    n_threads = fuel.get_rayon_parallelism()
    print(f"fuel {variant} finished in {elapsed:.3f} seconds n_threads={n_threads}")
    print(f"iterations={result.n_iter}, inertia={result.inertia}")
    return elapsed


def sample_init_centers(X, k, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(X.shape[0], size=k, replace=False)
    return X[indices]


def main():
    X = load_mnist(n_samples=10000)
    init_centers = sample_init_centers(X, k=10) #, seed=42)

    benchmark_sklearn(X, init_centers, algorithm="lloyd")
    benchmark_sklearn(X, init_centers, algorithm="elkan")
    benchmark_fuel(X, init_centers, variant="simp_elkan")
    benchmark_fuel(X, init_centers, variant="simp_hamerly")
    benchmark_fuel(X, init_centers, variant="shallot")
    benchmark_fuel(X, init_centers, variant="lloyd_blas")
    benchmark_fuel(X, init_centers, variant="lloyd")
    benchmark_fuel(X, init_centers, variant="lloyd_naive")


if __name__ == "__main__":
    main()
