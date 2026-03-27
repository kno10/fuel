use std::collections::BinaryHeap;
use std::env;
use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use csv::ReaderBuilder;
// additional imports from submodules
use fuel::Float;
use fuel::VectorData;
use fuel::api::{Data, DistanceData, DistanceSearch};
use fuel::covertree::CoverTree;
use fuel::data::TableQuery;
use fuel::distance::{DistanceFunction, Euclidean, PartialDistance};
use fuel::kd::{KdTree, MaxVarianceSplit};
use fuel::vptree::VPTree;
// TableWithDistance is available at the crate root for convenience
use fuel::{
    CoordinateQuery, IndexQuery, KnnSearch, PrioritySearcher, PrioritySearcherFactory, RangeSearch,
    TableWithDistance,
};
use rand::distributions::Standard;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() -> Result<(), Box<dyn Error>> {
    let mut dims = 2_usize;
    let mut n_points = 100_000_usize;
    let mut num_queries = 10_000_usize;
    let mut seed = 42_u64;
    let mut csv_path: Option<String> = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dims" => {
                dims = args.next().ok_or("missing value for --dims")?.parse()?;
            }
            "--npoints" => {
                n_points = args.next().ok_or("missing value for --npoints")?.parse()?;
            }
            "--queries" => {
                num_queries = args.next().ok_or("missing value for --queries")?.parse()?;
            }
            "--seed" => {
                seed = args.next().ok_or("missing value for --seed")?.parse()?;
            }
            "--csv" => {
                csv_path = Some(args.next().ok_or("missing value for --csv")?);
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            unknown => return Err(format!("unknown argument: {}", unknown).into()),
        }
    }

    let points = if let Some(path) = &csv_path {
        load_points_from_csv(path)?
    } else {
        let mut rng = StdRng::seed_from_u64(seed);
        generate_points(n_points, dims, &mut rng)
    };

    let point_dims = points.first().map(|p| p.len()).unwrap_or(0);
    let source = csv_path.as_deref().unwrap_or("random");

    if points.len() <= 1 {
        return Err("dataset must contain at least two points for benchmarking".into());
    }

    let mut rng = StdRng::seed_from_u64(42);
    let queries = (0..num_queries).map(|_| rng.gen_range(0..points.len())).collect::<Vec<_>>();

    let explore_count = ((points.len() as f64) * 0.01).ceil() as usize;
    let explore_count = explore_count.max(1).min(points.len());
    let neighbor_rank = explore_count.min(points.len() - 1);
    // Use the same query set for all methods (kd-tree, vp-tree, linear) for consistency.
    let explore_queries = queries.clone();

    println!("Dataset: {} points × {} dims (source: {})", points.len(), point_dims, source);
    println!(
        "Benchmark k={} (~10% of {} points), queries={}",
        neighbor_rank,
        points.len(),
        num_queries
    );

    let kd_metric = CountingPartialDistance::new(Euclidean);
    let kd_data = TableWithDistance::with_distance(&points, kd_metric.clone());
    kd_metric.reset();
    let kd_build_start = Instant::now();
    let kd_tree = KdTree::new(&kd_data, MaxVarianceSplit);
    let kd_build_time = kd_build_start.elapsed();
    let kd_build_distances = kd_metric.count();

    println!(
        "kd-tree  : build={:.3}s distances={}",
        kd_build_time.as_secs_f64(),
        kd_build_distances
    );

    let vp_distance = CountingDistance::new(Euclidean);
    let vp_data = TableWithDistance::with_distance(&points, vp_distance.clone());
    vp_distance.reset();
    let mut vp_rng = StdRng::seed_from_u64(1337);
    let vp_build_start = Instant::now();
    let vp_tree = VPTree::new(&vp_data, 10, &mut vp_rng);
    let vp_build_time = vp_build_start.elapsed();
    let vp_build_distances = vp_distance.count();
    println!(
        "vp-tree  : build={:.3}s distances={}",
        vp_build_time.as_secs_f64(),
        vp_build_distances
    );

    let ct_distance = CountingDistance::new(Euclidean);
    let ct_data = TableWithDistance::with_distance(&points, ct_distance.clone());
    let mut ct_rng = StdRng::seed_from_u64(2026);
    let ct_build_start = Instant::now();
    let ct_tree = CoverTree::new(&ct_data, 1.3, 1, &mut ct_rng);
    let ct_build_time = ct_build_start.elapsed();
    let ct_build_distances = ct_distance.count();

    print_build_report("kd-tree", kd_build_time, kd_build_distances);
    print_build_report("vp-tree", vp_build_time, vp_build_distances);
    print_build_report("cover-tree", ct_build_time, ct_build_distances);

    let (kd_knn_time, kd_knn_dist, kd_knn_avg) = measure_knn_coordinates(
        &kd_tree,
        &kd_data,
        &explore_queries,
        neighbor_rank,
        &kd_metric,
    );
    report_measure(
        "kNN kd-tree",
        explore_queries.len(),
        neighbor_rank,
        kd_knn_time,
        kd_knn_dist,
        kd_knn_avg,
        "avg-dist",
    );

    let vp_knn_distance = CountingDistance::new(Euclidean);
    let vp_knn_data = TableWithDistance::with_distance(&points, vp_knn_distance.clone());
    let (vp_knn_time, vp_knn_dist, vp_knn_avg) = measure_knn_index(
        &vp_tree,
        &vp_knn_data,
        &explore_queries,
        neighbor_rank,
        &vp_knn_distance,
    );
    report_measure(
        "kNN vp-tree",
        explore_queries.len(),
        neighbor_rank,
        vp_knn_time,
        vp_knn_dist,
        vp_knn_avg,
        "avg-dist",
    );

    let (ct_knn_time, ct_knn_dist, ct_knn_avg) = measure_knn_index(
        &ct_tree,
        &ct_data,
        &explore_queries,
        neighbor_rank,
        &ct_distance,
    );
    report_measure(
        "kNN cover-tree",
        explore_queries.len(),
        neighbor_rank,
        ct_knn_time,
        ct_knn_dist,
        ct_knn_avg,
        "avg-dist",
    );

    let range_radius = kd_knn_avg;
    let (kd_range_time, kd_range_dist, kd_range_avg) = measure_range_coordinates(
        &kd_tree,
        &kd_data,
        &explore_queries,
        range_radius,
        &kd_metric,
    );
    report_measure(
        "range kd-tree",
        explore_queries.len(),
        neighbor_rank,
        kd_range_time,
        kd_range_dist,
        kd_range_avg,
        "avg-results",
    );

    let ct_range_distance = CountingDistance::new(Euclidean);
    let ct_range_data = TableWithDistance::with_distance(&points, ct_range_distance.clone());
    let (ct_range_time, ct_range_dist, ct_range_avg) = measure_range_index(
        &ct_tree,
        &ct_range_data,
        &explore_queries,
        range_radius,
        &ct_range_distance,
    );
    report_measure(
        "range cover-tree",
        explore_queries.len(),
        neighbor_rank,
        ct_range_time,
        ct_range_dist,
        ct_range_avg,
        "avg-results",
    );

    let vp_range_distance = CountingDistance::new(Euclidean);
    let vp_range_data = TableWithDistance::with_distance(&points, vp_range_distance.clone());
    let (vp_range_time, vp_range_dist, vp_range_avg) = measure_range_index(
        &vp_tree,
        &vp_range_data,
        &explore_queries,
        range_radius,
        &vp_range_distance,
    );
    report_measure(
        "range vp-tree",
        explore_queries.len(),
        neighbor_rank,
        vp_range_time,
        vp_range_dist,
        vp_range_avg,
        "avg-results",
    );

    let kd_priority_metric = CountingPartialDistance::new(Euclidean);
    let (kd_priority_time, kd_priority_dist, kd_priority_avg) = measure_priority_coordinates(
        &kd_tree,
        &kd_data,
        &explore_queries,
        neighbor_rank,
        &kd_priority_metric,
    );
    report_measure(
        "priority kd-tree",
        explore_queries.len(),
        neighbor_rank,
        kd_priority_time,
        kd_priority_dist,
        kd_priority_avg,
        "avg-dist",
    );

    let vp_priority_distance = CountingDistance::new(Euclidean);
    let vp_priority_data = TableWithDistance::with_distance(&points, vp_priority_distance.clone());
    let (vp_priority_time, vp_priority_dist, vp_priority_avg) = measure_priority_index(
        &vp_tree,
        &vp_priority_data,
        &explore_queries,
        neighbor_rank,
        &vp_priority_distance,
    );
    report_measure(
        "priority vp-tree",
        explore_queries.len(),
        neighbor_rank,
        vp_priority_time,
        vp_priority_dist,
        vp_priority_avg,
        "avg-dist",
    );

    let ct_priority_distance = CountingDistance::new(Euclidean);
    let ct_priority_data = TableWithDistance::with_distance(&points, ct_priority_distance.clone());
    let (ct_priority_time, ct_priority_dist, ct_priority_avg) = measure_priority_index(
        &ct_tree,
        &ct_priority_data,
        &explore_queries,
        neighbor_rank,
        &ct_priority_distance,
    );
    report_measure(
        "priority cover-tree",
        explore_queries.len(),
        neighbor_rank,
        ct_priority_time,
        ct_priority_dist,
        ct_priority_avg,
        "avg-dist",
    );

    let linear_distance = CountingDistance::new(Euclidean);
    let linear_data = TableWithDistance::with_distance(&points, linear_distance.clone());
    let (linear_query_time, linear_dist, linear_avg) =
        measure_linear(&linear_data, &linear_distance, &queries, neighbor_rank);
    report_measure(
        "linear kNN",
        queries.len(),
        neighbor_rank,
        linear_query_time,
        linear_dist,
        linear_avg,
        "avg-dist",
    );

    Ok(())
}

fn generate_points(n: usize, dims: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let mut points = Vec::with_capacity(n);
    for _ in 0..n {
        let mut point = Vec::with_capacity(dims);
        for _ in 0..dims {
            point.push(rng.sample(Standard));
        }
        points.push(point);
    }
    points
}

trait Counter {
    fn reset(&self);
    fn count(&self) -> usize;
}

impl<D> Counter for CountingDistance<D> {
    fn reset(&self) { self.reset() }
    fn count(&self) -> usize { self.count() }
}

impl<M> Counter for CountingPartialDistance<M> {
    fn reset(&self) { self.reset() }
    fn count(&self) -> usize { self.count() }
}

fn kth_neighbor_distance_from_knn<T, Q>(
    tree: &T,
    query: &Q,
    query_idx: usize,
    rank: usize,
) -> Option<f64>
where
    T: KnnSearch<f64, Q>,
    Q: DistanceSearch<f64> + ?Sized,
{
    if rank == 0 {
        return None;
    }

    tree.search_knn(query, rank + 1)
        .into_iter()
        .filter(|neighbor| neighbor.index != query_idx)
        .nth(rank - 1)
        .map(|neighbor| neighbor.distance)
}

fn kth_neighbor_distance_from_searcher<Q, S>(
    searcher: &mut S,
    query: &Q,
    rank: usize,
    query_idx: usize,
) -> Option<f64>
where
    Q: DistanceSearch<f64> + ?Sized,
    S: PrioritySearcher<f64, Q>,
{
    if rank == 0 {
        return None;
    }

    let mut candidates: BinaryHeap<MaxDistance> = BinaryHeap::new();
    loop {
        if candidates.len() == rank
            && candidates
                .peek()
                .map(|worst| searcher.all_lower_bound() >= worst.0)
                .unwrap_or(false)
        {
            return candidates.peek().map(|worst| worst.0);
        }

        match searcher.next(query) {
            Some(neighbor) => {
                if neighbor.index == query_idx {
                    continue;
                }
                let dist = neighbor.distance;
                candidates.push(MaxDistance(dist));
                if candidates.len() > rank {
                    candidates.pop();
                }
            }
            None => return candidates.peek().map(|candidate| candidate.0),
        }
    }
}

fn measure_knn_index<'a, T, D, C>(
    tree: &T,
    data: &'a D,
    queries: &[usize],
    rank: usize,
    counter: &C,
) -> (std::time::Duration, usize, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + IndexQuery<f64>,
    T: KnnSearch<f64, D::Query<'a>>,
    C: Counter,
{
    counter.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    let mut query: D::Query<'a> = data.query();

    for &query_idx in queries {
        query.set_index(query_idx);
        if let Some(dist) = kth_neighbor_distance_from_knn(tree, &query, query_idx, rank) {
            sum += dist;
            found += 1;
        }
    }

    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn measure_knn_coordinates<'a, T, D, C>(
    tree: &T,
    data: &'a D,
    queries: &[usize],
    rank: usize,
    counter: &C,
) -> (std::time::Duration, usize, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64>,
    T: KnnSearch<f64, D::Query<'a>>,
    C: Counter,
{
    counter.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    let mut query: D::Query<'a> = data.query();

    for &query_idx in queries {
        query.set_coordinates(data.point(query_idx));
        if let Some(dist) = kth_neighbor_distance_from_knn(tree, &query, query_idx, rank) {
            sum += dist;
            found += 1;
        }
    }

    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn measure_range_index<'a, T, D, C>(
    tree: &T,
    data: &'a D,
    queries: &[usize],
    radius: f64,
    counter: &C,
) -> (std::time::Duration, usize, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + IndexQuery<f64>,
    T: RangeSearch<f64, D::Query<'a>>,
    C: Counter,
{
    counter.reset();
    let start = Instant::now();
    let mut total_found = 0_usize;
    let mut query: D::Query<'a> = data.query();

    for &query_idx in queries {
        query.set_index(query_idx);
        let neighbors = tree.search_range(&query, radius);
        total_found += neighbors.into_iter().filter(|neighbor| neighbor.index != query_idx).count();
    }

    let avg = if queries.is_empty() { 0.0 } else { total_found as f64 / queries.len() as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn measure_range_coordinates<'a, T, D, C>(
    tree: &T,
    data: &'a D,
    queries: &[usize],
    radius: f64,
    counter: &C,
) -> (std::time::Duration, usize, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64>,
    T: RangeSearch<f64, D::Query<'a>>,
    C: Counter,
{
    counter.reset();
    let start = Instant::now();
    let mut total_found = 0_usize;
    let mut query: D::Query<'a> = data.query();

    for &query_idx in queries {
        query.set_coordinates(data.point(query_idx));
        let neighbors = tree.search_range(&query, radius);
        total_found += neighbors.into_iter().filter(|neighbor| neighbor.index != query_idx).count();
    }

    let avg = if queries.is_empty() { 0.0 } else { total_found as f64 / queries.len() as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn measure_priority_index<'a, T, D, C>(
    tree: &T,
    data: &'a D,
    queries: &[usize],
    kth: usize,
    counter: &C,
) -> (std::time::Duration, usize, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + IndexQuery<f64>,
    T: PrioritySearcherFactory<f64, D::Query<'a>>,
    C: Counter,
{
    counter.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    let mut query: D::Query<'a> = data.query();

    for &query_idx in queries {
        query.set_index(query_idx);

        let mut searcher = <T as PrioritySearcherFactory<f64, D::Query<'_>>>::priority_searcher(tree);
        if let Some(dist) = kth_neighbor_distance_from_searcher(&mut searcher, &query, kth, query_idx) {
            sum += dist;
            found += 1;
        }
    }

    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn measure_priority_coordinates<'a, T, D, C>(
    tree: &T,
    data: &'a D,
    queries: &[usize],
    kth: usize,
    counter: &C,
) -> (std::time::Duration, usize, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64>,
    T: PrioritySearcherFactory<f64, D::Query<'a>>,
    C: Counter,
{
    counter.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    let mut query: D::Query<'a> = data.query();

    for &query_idx in queries {
        query.set_coordinates(data.point(query_idx));

        let mut searcher = <T as PrioritySearcherFactory<f64, D::Query<'_>>>::priority_searcher(tree);
        if let Some(dist) = kth_neighbor_distance_from_searcher(&mut searcher, &query, kth, query_idx) {
            sum += dist;
            found += 1;
        }
    }

    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn measure_linear(
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<Euclidean>, f64>,
    counter: &CountingDistance<Euclidean>, queries: &[usize], rank: usize,
) -> (std::time::Duration, usize, f64) {
    counter.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    for &query_idx in queries {
        if let Some(dist) = linear_kth_neighbor_distance(data, query_idx, rank) {
            sum += dist;
            found += 1;
        }
    }
    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn linear_kth_neighbor_distance(
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<Euclidean>, f64>,
    query_idx: usize, rank: usize,
) -> Option<f64> {
    if rank == 0 {
        return None;
    }
    let mut distances: Vec<(f64, usize)> =
        data.iter().map(|idx| (data.distance(query_idx, idx), idx)).collect();
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    distances.iter().filter(|(_, idx)| *idx != query_idx).nth(rank - 1).map(|(dist, _)| *dist)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MaxDistance(f64);

impl Eq for MaxDistance {}

impl PartialOrd for MaxDistance {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}

impl Ord for MaxDistance {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn load_points_from_csv(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(false).from_path(path)?;
    let mut points = Vec::new();
    let mut dims = None;
    let mut label_column = None;

    for record in reader.records() {
        let record = record?;
        if record.is_empty() {
            continue;
        }

        if dims.is_none() && record.iter().any(|field| field.parse::<f64>().is_err()) {
            for (idx, field) in record.iter().enumerate() {
                let lower = field.trim().to_ascii_lowercase();
                if lower.contains("label") || lower.contains("class") {
                    label_column = Some(idx);
                    break;
                }
            }
            continue;
        }

        let mut row = Vec::with_capacity(record.len());
        for (idx, field) in record.iter().enumerate() {
            if Some(idx) == label_column {
                continue;
            }
            let value = field.parse::<f64>().map_err(|e| format!("{}: {}", field, e))?;
            row.push(value);
        }

        if row.is_empty() {
            continue;
        }

        if let Some(expected) = dims {
            if row.len() != expected {
                return Err(format!(
                    "CSV row length {} differs from expected {}",
                    row.len(),
                    expected
                )
                .into());
            }
        } else {
            dims = Some(row.len());
        }

        points.push(row);
    }

    if points.is_empty() {
        return Err(format!("CSV file \"{}\" contains no data rows", path).into());
    }

    Ok(points)
}

fn print_help() {
    eprintln!("kd_vs_vp benchmark usage:");
    eprintln!("  kd_vs_vp [--dims N] [--npoints N] [--queries N] [--seed N] [--csv PATH]");
    eprintln!("Options:");
    eprintln!("  --dims N       Number of dimensions (default: 2)");
    eprintln!("  --npoints N    Number of points to generate (default: 100000)");
    eprintln!("  --queries N    Number of query points (default: 10000)");
    eprintln!("  --seed N       RNG seed (default: 42)");
    eprintln!("  --csv PATH     Load points from CSV instead of generating random data");
    eprintln!("  --help, -h     Show this help message");
}

fn print_build_report(name: &str, elapsed: std::time::Duration, distance_calls: usize) {
    println!(
        "{:10}: build={:.3}s distances={}",
        name,
        elapsed.as_secs_f64(),
        distance_calls
    );
}

fn report_measure(
    name: &str,
    queries: usize,
    k: usize,
    elapsed: std::time::Duration,
    distance_calls: usize,
    avg: f64,
    avg_label: &str,
) {
    println!(
        "{} (queries={}, k={}) : query={:.3}s distances={} {}={:.6}",
        name,
        queries,
        k,
        elapsed.as_secs_f64(),
        distance_calls,
        avg_label,
        avg
    );
}

#[derive(Debug)]
struct CountingDistance<D> {
    inner: D,
    counter: Arc<AtomicUsize>,
}

impl<D> CountingDistance<D> {
    fn new(inner: D) -> Self { Self { inner, counter: Arc::new(AtomicUsize::new(0)) } }

    fn count(&self) -> usize { self.counter.load(Ordering::Relaxed) }

    fn reset(&self) { self.counter.store(0, Ordering::Relaxed); }
}

impl<D: Clone> Clone for CountingDistance<D> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), counter: Arc::clone(&self.counter) }
    }
}

impl<D, T: ?Sized, F> DistanceFunction<T, F> for CountingDistance<D>
where
    D: DistanceFunction<T, F>,
    F: Float,
{
    fn distance(&self, a: &T, b: &T) -> F {
        self.counter.fetch_add(1, Ordering::Relaxed);
        self.inner.distance(a, b)
    }
}

#[derive(Debug)]
struct CountingPartialDistance<M> {
    inner: M,
    counter: Arc<AtomicUsize>,
}

impl<M> CountingPartialDistance<M> {
    fn new(inner: M) -> Self { Self { inner, counter: Arc::new(AtomicUsize::new(0)) } }

    fn count(&self) -> usize { self.counter.load(Ordering::Relaxed) }

    fn reset(&self) { self.counter.store(0, Ordering::Relaxed); }
}

impl<M: Clone> Clone for CountingPartialDistance<M> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), counter: Arc::clone(&self.counter) }
    }
}

impl<M, F> PartialDistance<F, F> for CountingPartialDistance<M>
where
    M: PartialDistance<F, F>,
    F: Float,
{
    fn axis_distance(&self, delta: F) -> F {
        self.counter.fetch_add(1, Ordering::Relaxed);
        self.inner.axis_distance(delta)
    }

    fn combine_axis_distances(&self, a: F, b: F) -> F { self.inner.combine_axis_distances(a, b) }
}

impl<M, F> DistanceFunction<[F], F> for CountingPartialDistance<M>
where
    M: DistanceFunction<[F], F>,
    F: Float,
{
    fn distance(&self, a: &[F], b: &[F]) -> F {
        self.counter.fetch_add(1, Ordering::Relaxed);
        self.inner.distance(a, b)
    }
}

impl<M, F> DistanceFunction<Vec<F>, F> for CountingPartialDistance<M>
where
    M: DistanceFunction<[F], F>,
    F: Float,
{
    fn distance(&self, a: &Vec<F>, b: &Vec<F>) -> F {
        self.counter.fetch_add(1, Ordering::Relaxed);
        self.inner.distance(a.as_slice(), b.as_slice())
    }
}
