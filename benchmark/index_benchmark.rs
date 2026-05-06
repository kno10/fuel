use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::time::Instant;

mod common;
use common::{CountingDistance, generate_points, load_points_from_csv};
use fuel::api::{Data, DistanceData, DistanceSearch};
use fuel::distance::{DistanceFunction, Euclidean};
use fuel::search::covertree::{CoverTree, expansion_heuristic_from_id};
use fuel::search::kdtree::{KdTree, MaxVarianceSplit};
use fuel::search::vptree::VPTree;
use fuel::{
    CoordinateQuery, DistPair, IndexQuery, KNNHeap, KnnSearch, PrioritySearcher,
    PrioritySearcherFactory, RangeSearch, TableWithDistance, VectorData,
};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

fn main() -> Result<(), Box<dyn Error>> {
    let mut dims = 2_usize;
    let mut n_points = 100_000_usize;
    let mut num_queries = 10_000_usize;
    let mut seed = 42_u64;
    let mut csv_path: Option<String> = None;

    let mut selected_trees: HashSet<String> =
        ["kd", "vp", "ct"].iter().map(|s| s.to_string()).collect();

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
            "--trees" => {
                let trees = args.next().ok_or("missing value for --trees")?;
                let parsed = trees
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<HashSet<_>>();
                for tree in &parsed {
                    if !["kd", "vp", "ct"].contains(&tree.as_str()) {
                        return Err(format!("unknown tree: {}", tree).into());
                    }
                }
                if parsed.is_empty() {
                    return Err("--trees value must include at least one of kd,vp,ct".into());
                }
                selected_trees = parsed;
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
    let queries = (0..num_queries).map(|_| rng.random_range(0..points.len())).collect::<Vec<_>>();

    let k = ((points.len() as f64).sqrt() * 0.1).ceil() as usize;
    println!("Dataset: {} points × {} dims (source: {})", points.len(), point_dims, source);
    println!(
        "Benchmark k={} (~{:.3}% of {} points), queries={}",
        k,
        k as f64 / points.len() as f64 * 100.0,
        points.len(),
        num_queries
    );

    let should_run_kd = selected_trees.contains("kd");
    let should_run_vp = selected_trees.contains("vp");
    let should_run_ct = selected_trees.contains("ct");

    if !should_run_kd && !should_run_vp && !should_run_ct {
        return Err("no indexed tree selected (use --trees)".into());
    }

    let kd_tree = if should_run_kd {
        let kd_metric = CountingDistance::new(Euclidean);
        let kd_data: TableWithDistance<'_, _, _, _, f64> =
            TableWithDistance::with_distance(&points, kd_metric.clone());
        let kd_build_start = Instant::now();
        let kd_tree = KdTree::new(&kd_data, MaxVarianceSplit);
        let kd_build_time = kd_build_start.elapsed();
        let kd_build_distances = kd_metric.count();
        print_build_report("kd-tree", kd_build_time, kd_build_distances);
        Some(kd_tree)
    } else {
        None
    };

    let vp_tree = if should_run_vp {
        let vp_distance = CountingDistance::new(Euclidean);
        let vp_data = TableWithDistance::with_distance(&points, vp_distance.clone());
        let mut vp_rng = StdRng::seed_from_u64(seed);
        let vp_build_start = Instant::now();
        let vp_tree = VPTree::new(&vp_data, 10, &mut vp_rng);
        let vp_build_time = vp_build_start.elapsed();
        let vp_build_distances = vp_distance.count();
        print_build_report("vp-tree", vp_build_time, vp_build_distances);
        Some(vp_tree)
    } else {
        None
    };

    let ct_tree = if should_run_ct {
        let ct_distance = CountingDistance::new(Euclidean);
        let ct_data = TableWithDistance::with_distance(&points, ct_distance.clone());
        let _ct_rng = StdRng::seed_from_u64(seed);
        let ct_build_start = Instant::now();
        let ct_tree = CoverTree::new(&ct_data, expansion_heuristic_from_id(dims as f64), 0);
        let ct_build_time = ct_build_start.elapsed();
        let ct_build_distances = ct_distance.count();
        print_build_report("cover-tree", ct_build_time, ct_build_distances);
        Some(ct_tree)
    } else {
        None
    };

    let mut range_radius = 0.0;

    if let Some(ref kd_tree) = kd_tree {
        let kd_knn_metric = CountingDistance::new(Euclidean);
        range_radius = run_knn(
            "kNN kd-tree",
            kd_tree,
            &TableWithDistance::with_distance(&points, kd_knn_metric.clone()),
            &queries,
            k,
            &kd_knn_metric,
        );
    }

    if let Some(ref vp_tree) = vp_tree {
        let vp_knn_metric = CountingDistance::new(Euclidean);
        range_radius = run_knn(
            "kNN vp-tree",
            vp_tree,
            &TableWithDistance::with_distance(&points, vp_knn_metric.clone()),
            &queries,
            k,
            &vp_knn_metric,
        );
    }

    if let Some(ref ct_tree) = ct_tree {
        let ct_knn_metric = CountingDistance::new(Euclidean);
        range_radius = run_knn(
            "kNN cover-tree",
            ct_tree,
            &TableWithDistance::with_distance(&points, ct_knn_metric.clone()),
            &queries,
            k,
            &ct_knn_metric,
        );
    }

    if let Some(ref kd_tree) = kd_tree {
        let kd_range_metric = CountingDistance::new(Euclidean);
        run_range(
            "range kd-tree",
            kd_tree,
            &TableWithDistance::with_distance(&points, kd_range_metric.clone()),
            &queries,
            range_radius,
            &kd_range_metric,
        );
    }

    if let Some(ref vp_tree) = vp_tree {
        let vp_range_metric = CountingDistance::new(Euclidean);
        run_range(
            "range vp-tree",
            vp_tree,
            &TableWithDistance::with_distance(&points, vp_range_metric.clone()),
            &queries,
            range_radius,
            &vp_range_metric,
        );
    }

    if let Some(ref ct_tree) = ct_tree {
        let ct_range_metric = CountingDistance::new(Euclidean);
        run_range(
            "range cover-tree",
            ct_tree,
            &TableWithDistance::with_distance(&points, ct_range_metric.clone()),
            &queries,
            range_radius,
            &ct_range_metric,
        );
    }

    if let Some(ref kd_tree) = kd_tree {
        let kd_priority_metric = CountingDistance::new(Euclidean);
        run_priority(
            "priority kd-tree",
            kd_tree,
            &TableWithDistance::with_distance(&points, kd_priority_metric.clone()),
            &queries,
            k,
            &kd_priority_metric,
        );
    }

    if let Some(ref vp_tree) = vp_tree {
        let vp_priority_metric = CountingDistance::new(Euclidean);
        run_priority(
            "priority vp-tree",
            vp_tree,
            &TableWithDistance::with_distance(&points, vp_priority_metric.clone()),
            &queries,
            k,
            &vp_priority_metric,
        );
    }

    if let Some(ref ct_tree) = ct_tree {
        let ct_priority_metric = CountingDistance::new(Euclidean);
        run_priority(
            "priority cover-tree",
            ct_tree,
            &TableWithDistance::with_distance(&points, ct_priority_metric.clone()),
            &queries,
            k,
            &ct_priority_metric,
        );
    }

    {
        let linear_metric = CountingDistance::new(Euclidean);
        let linear_data = TableWithDistance::with_distance(&points, linear_metric.clone());
        run_linear("linear kNN", &linear_data, &queries, k, &linear_metric);
    }

    Ok(())
}

fn kth_neighbor_distance_from_searcher<Q, S>(
    searcher: &mut S, query: &Q, rank: usize,
) -> Option<f64>
where
    Q: DistanceSearch<f64> + ?Sized,
    S: PrioritySearcher<f64, Q>,
{
    let mut knn = KNNHeap::new(rank);
    while let Some(neighbor) = searcher.next(query) {
        knn.insert(DistPair::new(neighbor.distance, neighbor.index));
        if knn.len() == rank {
            let k_distance = knn.k_distance();
            searcher.decrease_cutoff(k_distance);
            if searcher.all_lower_bound() >= k_distance {
                return Some(k_distance);
            }
        }
    }

    if knn.len() >= rank {
        return Some(knn.k_distance());
    }
    let mut candidates = knn.into_vec();
    candidates.sort_unstable();
    candidates.last().map(|candidate| candidate.distance)
}

fn measure_knn<'a, T, D>(
    tree: &T, data: &'a D, queries: &[usize], rank: usize,
) -> (std::time::Duration, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64> + IndexQuery<f64>,
    T: KnnSearch<f64, D::Query<'a>>,
{
    let start = Instant::now();
    let (mut sum, mut found) = (0.0, 0);
    let query: D::Query<'a> = data.query();

    for &query_idx in queries {
        if let Some(dist) = {
            tree.search_knn(&query.with_coordinates(data.point(query_idx)), rank)
                .into_iter()
                .nth(rank - 1)
                .map(|neighbor| neighbor.distance)
        } {
            sum += dist;
            found += 1;
        }
    }
    (start.elapsed(), if found == 0 { 0.0 } else { sum / found as f64 })
}

fn measure_range<'a, T, D>(
    tree: &T, data: &'a D, queries: &[usize], radius: f64,
) -> (std::time::Duration, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64> + IndexQuery<f64>,
    T: RangeSearch<f64, D::Query<'a>>,
{
    let start = Instant::now();
    let mut total_found = 0_usize;
    let query: D::Query<'a> = data.query();

    for &query_idx in queries {
        let neighbors = tree.search_range(&query.with_index(query_idx), radius);
        total_found += neighbors.into_iter().filter(|neighbor| neighbor.index != query_idx).count();
    }
    (
        start.elapsed(),
        if queries.is_empty() { 0.0 } else { total_found as f64 / queries.len() as f64 },
    )
}

fn measure_priority<'a, T, D>(
    tree: &T, data: &'a D, queries: &[usize], kth: usize,
) -> (std::time::Duration, f64)
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64> + IndexQuery<f64>,
    T: PrioritySearcherFactory<f64, D::Query<'a>>,
{
    let start = Instant::now();
    let (mut sum, mut found) = (0.0, 0);
    let query: D::Query<'a> = data.query();

    for &query_idx in queries {
        let mut searcher = <T as PrioritySearcherFactory<f64, _>>::priority_searcher(tree);
        if let Some(dist) = kth_neighbor_distance_from_searcher(&mut searcher, &query.with_index(query_idx), kth) {
            sum += dist;
            found += 1;
        }
    }
    (start.elapsed(), if found == 0 { 0.0 } else { sum / found as f64 })
}

fn run_knn<'a, T, D>(
    name: &str, tree: &T, data: &'a D, queries: &[usize], rank: usize,
    metric: &CountingDistance<Euclidean>,
) -> f64
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64> + IndexQuery<f64>,
    T: KnnSearch<f64, D::Query<'a>>,
{
    let (time, avg) = measure_knn(tree, data, queries, rank);
    report_measure(name, queries.len(), rank, time, metric.count(), avg);
    avg
}

fn run_range<'a, T, D>(
    name: &str, tree: &T, data: &'a D, queries: &[usize], radius: f64,
    metric: &CountingDistance<Euclidean>,
) -> f64
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64> + IndexQuery<f64>,
    T: RangeSearch<f64, D::Query<'a>>,
{
    let (time, avg) = measure_range(tree, data, queries, radius);
    report_range_measure(name, queries.len(), radius, time, metric.count(), avg);
    avg
}

fn run_priority<'a, T, D>(
    name: &str, tree: &T, data: &'a D, queries: &[usize], k: usize,
    metric: &CountingDistance<Euclidean>,
) -> f64
where
    D: DistanceData<f64> + VectorData<f64> + 'a,
    D::Query<'a>: DistanceSearch<f64> + CoordinateQuery<f64, f64> + IndexQuery<f64>,
    T: PrioritySearcherFactory<f64, D::Query<'a>>,
{
    let (time, avg) = measure_priority(tree, data, queries, k);
    report_measure(name, queries.len(), k, time, metric.count(), avg);
    avg
}

fn run_linear<'a, C>(
    name: &str, data: &TableWithDistance<'a, f64, Vec<f64>, C, f64>, queries: &[usize],
    rank: usize, metric: &CountingDistance<Euclidean>,
) -> f64
where
    C: DistanceFunction<[f64], f64>,
{
    let start = Instant::now();
    let (mut sum, mut found) = (0.0, 0);
    for &query_idx in queries {
        if let Some(dist) = linear_kth_neighbor_distance(data, query_idx, rank) {
            sum += dist;
            found += 1;
        }
    }
    let (time, avg) = (start.elapsed(), if found == 0 { 0.0 } else { sum / found as f64 });
    report_measure(name, queries.len(), rank, time, metric.count(), avg);
    avg
}

fn linear_kth_neighbor_distance<'a, C>(
    data: &TableWithDistance<'a, f64, Vec<f64>, C, f64>, query_idx: usize, rank: usize,
) -> Option<f64>
where
    C: DistanceFunction<[f64], f64>,
{
    let mut heap = KNNHeap::new(rank);
    // including query point for consistency
    for idx in data.iter() {
        heap.insert(DistPair::new(data.distance(query_idx, idx), idx));
    }

    if heap.len() < rank {
        return None;
    }
    Some(heap.k_distance())
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
    println!("{:10}: build={:.3}s distances={}", name, elapsed.as_secs_f64(), distance_calls);
}

fn report_measure(
    name: &str, queries: usize, k: usize, elapsed: std::time::Duration, distance_calls: usize,
    avg: f64,
) {
    println!(
        "{:<20} (queries={}, k={}) : query={:.3}s distances={} avg-dist={:.8}",
        name,
        queries,
        k,
        elapsed.as_secs_f64(),
        distance_calls,
        avg
    );
}

fn report_range_measure(
    name: &str, queries: usize, radius: f64, elapsed: std::time::Duration, distance_calls: usize,
    avg: f64,
) {
    println!(
        "{:<20} (queries={}, radius={:.6e}) : query={:.3}s distances={} avg-results={:.6}",
        name,
        queries,
        radius,
        elapsed.as_secs_f64(),
        distance_calls,
        avg
    );
}
