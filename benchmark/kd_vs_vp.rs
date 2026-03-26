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
use fuel::data::TableQuery;
use fuel::distance::{DistanceFunction, EuclideanDistance, PartialDistance};
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

    let kd_metric = CountingPartialDistance::new(EuclideanDistance);
    let kd_data = TableWithDistance::with_distance(&points, kd_metric.clone());
    kd_metric.reset();
    let kd_build_start = Instant::now();
    let kd_tree = KdTree::new(&kd_data, MaxVarianceSplit);
    let kd_build_time = kd_build_start.elapsed();
    let kd_build_distances = kd_metric.count();

    let vp_distance = CountingDistance::new(EuclideanDistance);
    let vp_data = TableWithDistance::with_distance(&points, vp_distance.clone());
    vp_distance.reset();
    let mut vp_rng = StdRng::seed_from_u64(1337);
    let vp_build_start = Instant::now();
    let vp_tree = VPTree::new(&vp_data, 10, &mut vp_rng);
    let vp_build_time = vp_build_start.elapsed();
    let vp_build_distances = vp_distance.count();

    println!("Dataset: {} points × {} dims (source: {})", points.len(), point_dims, source);
    println!(
        "Benchmark k={} (~10% of {} points), queries={}",
        neighbor_rank,
        points.len(),
        num_queries
    );
    println!(
        "kd-tree  : build={:.3}s distances={}",
        kd_build_time.as_secs_f64(),
        kd_build_distances
    );
    println!(
        "vp-tree  : build={:.3}s distances={}",
        vp_build_time.as_secs_f64(),
        vp_build_distances
    );

    let (kd_knn_time, kd_knn_dist, kd_knn_avg) =
        measure_kd_kth_neighbor(&kd_tree, &kd_data, &explore_queries, neighbor_rank, &kd_metric);
    println!(
        "kNN kd-tree (queries={}, k={}) : query={:.3}s distances={} avg-dist={:.6}",
        explore_queries.len(),
        neighbor_rank,
        kd_knn_time.as_secs_f64(),
        kd_knn_dist,
        kd_knn_avg
    );

    let vp_knn_distance = CountingDistance::new(EuclideanDistance);
    let vp_knn_data = TableWithDistance::with_distance(&points, vp_knn_distance.clone());
    let (vp_knn_time, vp_knn_dist, vp_knn_avg) = measure_vp_kth_neighbor(
        &vp_tree,
        &vp_knn_data,
        &explore_queries,
        neighbor_rank,
        &vp_knn_distance,
    );
    println!(
        "kNN vp-tree (queries={}, k={}) : query={:.3}s distances={} avg-dist={:.6}",
        explore_queries.len(),
        neighbor_rank,
        vp_knn_time.as_secs_f64(),
        vp_knn_dist,
        vp_knn_avg
    );

    let kd_range_radius = kd_knn_avg.max(0.0);
    let (kd_range_time, kd_range_dist, kd_range_avg) =
        measure_kd_range(&kd_tree, &kd_data, &explore_queries, kd_range_radius, &kd_metric);
    println!(
        "range kd-tree (radius={:.6}, queries={}, k={}) : query={:.3}s distances={} avg-results={:.3}",
        kd_range_radius,
        explore_queries.len(),
        neighbor_rank,
        kd_range_time.as_secs_f64(),
        kd_range_dist,
        kd_range_avg
    );

    let vp_range_radius = vp_knn_avg.max(0.0);
    let vp_range_distance = CountingDistance::new(EuclideanDistance);
    let vp_range_data = TableWithDistance::with_distance(&points, vp_range_distance.clone());
    let (vp_range_time, vp_range_dist, vp_range_avg) = measure_vp_range(
        &vp_tree,
        &vp_range_data,
        &explore_queries,
        vp_range_radius,
        &vp_range_distance,
    );
    println!(
        "range vp-tree (radius={:.6}, queries={}, k={}) : query={:.3}s distances={} avg-results={:.3}",
        vp_range_radius,
        explore_queries.len(),
        neighbor_rank,
        vp_range_time.as_secs_f64(),
        vp_range_dist,
        vp_range_avg
    );

    let kd_priority_metric = CountingPartialDistance::new(EuclideanDistance);
    let (kd_priority_time, kd_priority_dist, kd_priority_avg) = measure_kd_priority(
        &kd_tree,
        &kd_data,
        &explore_queries,
        &kd_priority_metric,
        neighbor_rank,
    );
    println!(
        "priority kd-tree (queries={}, k={}) : query={:.3}s distances={} avg-dist={:.6}",
        explore_queries.len(),
        neighbor_rank,
        kd_priority_time.as_secs_f64(),
        kd_priority_dist,
        kd_priority_avg
    );

    let vp_priority_distance = CountingDistance::new(EuclideanDistance);
    let vp_priority_data = TableWithDistance::with_distance(&points, vp_priority_distance.clone());
    let (vp_priority_time, vp_priority_dist, vp_priority_avg) = measure_vp_priority(
        &vp_tree,
        &vp_priority_data,
        &explore_queries,
        &vp_priority_distance,
        neighbor_rank,
    );
    println!(
        "priority vp-tree (queries={}, k={}) : query={:.3}s distances={} avg-dist={:.6}",
        explore_queries.len(),
        neighbor_rank,
        vp_priority_time.as_secs_f64(),
        vp_priority_dist,
        vp_priority_avg
    );

    let linear_distance = CountingDistance::new(EuclideanDistance);
    let linear_data = TableWithDistance::with_distance(&points, linear_distance.clone());
    let (linear_query_time, linear_dist, linear_avg) =
        measure_linear(&linear_data, &linear_distance, &queries, neighbor_rank);
    println!(
        "linear kNN : query={:.3}s distances={} avg-dist={:.6}",
        linear_query_time.as_secs_f64(),
        linear_dist,
        linear_avg
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

fn measure_kd_kth_neighbor(
    tree: &KdTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingPartialDistance<EuclideanDistance>, f64>,
    queries: &[usize], rank: usize, metric: &CountingPartialDistance<EuclideanDistance>,
) -> (std::time::Duration, usize, f64) {
    metric.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    let mut query = data.query();
    for &query_idx in queries {
        if let Some(dist) = kth_neighbor_distance_kd(tree, data, &mut query, query_idx, rank) {
            sum += dist;
            found += 1;
        }
    }
    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), metric.count(), avg)
}

fn kth_neighbor_distance_kd(
    tree: &KdTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingPartialDistance<EuclideanDistance>, f64>,
    query: &mut impl CoordinateQuery<f64, f64>, query_idx: usize, rank: usize,
) -> Option<f64> {
    if rank == 0 {
        return None;
    }
    query.set_coordinates(data.point(query_idx));
    let neighbors = tree.search_knn(query, rank + 1);
    neighbors
        .into_iter()
        .filter(|neighbor| neighbor.index != query_idx)
        .nth(rank - 1)
        .map(|neighbor| neighbor.distance)
}

fn measure_vp_kth_neighbor(
    tree: &VPTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<EuclideanDistance>, f64>,
    queries: &[usize], rank: usize, counter: &CountingDistance<EuclideanDistance>,
) -> (std::time::Duration, usize, f64) {
    counter.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    for &query_idx in queries {
        if let Some(dist) = kth_neighbor_distance_vp(tree, data, query_idx, rank) {
            sum += dist;
            found += 1;
        }
    }
    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn kth_neighbor_distance_vp(
    tree: &VPTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<EuclideanDistance>, f64>,
    query_idx: usize, rank: usize,
) -> Option<f64> {
    if rank == 0 {
        return None;
    }
    let query = data.query().with_index(query_idx);
    tree.search_knn(&query, rank + 1)
        .into_iter()
        .filter(|neighbor| neighbor.index != query_idx)
        .nth(rank - 1)
        .map(|neighbor| neighbor.distance)
}

fn measure_linear(
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<EuclideanDistance>, f64>,
    counter: &CountingDistance<EuclideanDistance>, queries: &[usize], rank: usize,
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

fn measure_kd_range(
    tree: &KdTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingPartialDistance<EuclideanDistance>, f64>,
    queries: &[usize], radius: f64, metric: &CountingPartialDistance<EuclideanDistance>,
) -> (std::time::Duration, usize, f64) {
    metric.reset();
    let start = Instant::now();
    let mut total_found = 0usize;
    let mut query = data.query();
    for &query_idx in queries {
        query.set_coordinates(data.point(query_idx));
        let neighbors = tree.search_range(&query, radius);
        total_found += neighbors.into_iter().filter(|neighbor| neighbor.index != query_idx).count();
    }
    let avg = if queries.is_empty() { 0.0 } else { total_found as f64 / queries.len() as f64 };
    (start.elapsed(), metric.count(), avg)
}

fn measure_vp_range(
    tree: &VPTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<EuclideanDistance>, f64>,
    queries: &[usize], radius: f64, counter: &CountingDistance<EuclideanDistance>,
) -> (std::time::Duration, usize, f64) {
    counter.reset();
    let start = Instant::now();
    let mut total_found = 0usize;
    let mut query = data.query();
    for &query_idx in queries {
        let mut count = 0usize;
        query.set_index(query_idx);
        tree.search_range(&query, radius, |pair| {
            if pair.index != query_idx {
                count += 1;
            }
        });
        total_found += count;
    }
    let avg = if queries.is_empty() { 0.0 } else { total_found as f64 / queries.len() as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn linear_kth_neighbor_distance(
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<EuclideanDistance>, f64>,
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

fn measure_kd_priority(
    tree: &KdTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingPartialDistance<EuclideanDistance>, f64>,
    explore_queries: &[usize], metric: &CountingPartialDistance<EuclideanDistance>, kth: usize,
) -> (std::time::Duration, usize, f64) {
    metric.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut query = data.query();
    for &query_idx in explore_queries {
        let query_point = data.point(query_idx);
        query.set_coordinates(query_point);
        let mut rank = 0;
        let mut distance = None;
        let mut searcher = <KdTree<f64> as PrioritySearcherFactory<
            f64,
            TableQuery<'_, '_, f64, Vec<f64>, CountingPartialDistance<EuclideanDistance>, f64>,
        >>::priority_searcher(tree);
        loop {
            if rank >= kth {
                break;
            }
            let Some(neighbor) = searcher.next(&query) else {
                break;
            };
            if neighbor.index == query_idx {
                continue;
            }
            rank += 1;
            if rank == kth {
                distance = Some(neighbor.distance);
                break;
            }
        }
        if let Some(dist) = distance {
            sum += dist;
        }
    }
    let avg = if explore_queries.is_empty() { 0.0 } else { sum / explore_queries.len() as f64 };
    (start.elapsed(), metric.count(), avg)
}

fn measure_vp_priority(
    tree: &VPTree<f64>,
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<EuclideanDistance>, f64>,
    explore_queries: &[usize], counter: &CountingDistance<EuclideanDistance>, kth: usize,
) -> (std::time::Duration, usize, f64) {
    counter.reset();
    let start = Instant::now();
    let mut sum = 0.0;
    let mut found = 0;
    let mut query = data.query();
    for &query_idx in explore_queries {
        query.set_index(query_idx);
        let mut searcher = <VPTree<f64> as PrioritySearcherFactory<
            f64,
            TableQuery<'_, '_, f64, Vec<f64>, CountingDistance<EuclideanDistance>, f64>,
        >>::priority_searcher(tree);
        if let Some(dist) =
            kth_neighbor_distance_from_vp_searcher(&mut searcher, &query, kth, query_idx)
        {
            sum += dist;
            found += 1;
        }
    }
    let avg = if found == 0 { 0.0 } else { sum / found as f64 };
    (start.elapsed(), counter.count(), avg)
}

fn kth_neighbor_distance_from_vp_searcher<Q, S>(
    searcher: &mut S, data: &Q, rank: usize, query_idx: usize,
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
            && candidates.peek().map(|worst| searcher.all_lower_bound() >= worst.0).unwrap_or(false)
        {
            return candidates.peek().map(|worst| worst.0);
        }

        match searcher.next(data) {
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
