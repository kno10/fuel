use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::time::Instant;

mod common;

use common::{CountingDistance, generate_points, load_points_from_csv};
use fuel::api::ApproxKnnSearch;
use fuel::distance::Euclidean;
use fuel::kd::{KdTree, MaxVarianceSplit};
use fuel::{DistanceData, IndexQuery, KnnSearch, TableWithDistance};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn main() -> Result<(), Box<dyn Error>> {
    let mut dims = 2_usize;
    let mut n_points = 100_000_usize;
    let mut num_queries = 10_000_usize;
    let mut seed = 42_u64;
    let mut csv_path: Option<String> = None;
    let mut rate_list = vec![0.001f32, 0.002, 0.005];

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
            "--rates" => {
                let rates = args.next().ok_or("missing value for --rates")?;
                rate_list = rates
                    .split(',')
                    .map(|s| s.trim().parse::<f32>())
                    .collect::<Result<Vec<_>, _>>()?;
                if rate_list.is_empty() {
                    return Err("--rates must include at least one value".into());
                }
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
        return Err("dataset must contain at least two points".into());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let query_indices =
        (0..num_queries).map(|_| rng.gen_range(0..points.len())).collect::<Vec<_>>();

    let k = ((points.len() as f64).sqrt() * 0.1).ceil() as usize;
    println!("Dataset: {} points × {} dims (source: {})", points.len(), point_dims, source);
    println!(
        "Benchmark k={} (~{:.3}% of {} points), queries={}",
        k,
        k as f64 / points.len() as f64 * 100.0,
        points.len(),
        num_queries
    );
    println!("Rates: {:?}", rate_list);

    // exact kd-tree reference (no counting)
    let exact_data = TableWithDistance::with_distance(&points, Euclidean);
    let exact_tree = KdTree::new(&exact_data, MaxVarianceSplit);

    // Precompute exact neighbors for accuracy
    let exact_neighbors_per_query = query_indices
        .iter()
        .map(|&query_idx| {
            let mut q = exact_data.query();
            q.set_index(query_idx);
            exact_tree.search_knn(&q, k).into_iter().map(|dp| dp.index).collect::<HashSet<_>>()
        })
        .collect::<Vec<_>>();

    for &rate in &rate_list {
        let metric = CountingDistance::new(Euclidean);
        let data = TableWithDistance::with_distance(&points, metric.clone());
        let tree = KdTree::new(&data, MaxVarianceSplit);

        let start = Instant::now();
        let mut total_recall = 0.0;

        for (query_id, &query_idx) in query_indices.iter().enumerate() {
            let mut q = data.query();
            q.set_index(query_idx);

            let approx = tree.search_aknn(&q, k, rate);
            let approx_idx: HashSet<usize> = approx.into_iter().map(|dp| dp.index).collect();
            let exact_idx = &exact_neighbors_per_query[query_id];

            let overlap = exact_idx.intersection(&approx_idx).count();
            total_recall += (overlap as f64) / (exact_idx.len() as f64);
        }

        let duration = start.elapsed();
        let total_distances = metric.count();
        let query_count = query_indices.len() as f64;
        let distances_rate = total_distances as f64 / query_count / points.len() as f64;
        let average_recall = total_recall / query_count;

        println!(
            "akNN (rate={:.3}) : time={:.3}s rate={:.7} distances={} recall={:.5}",
            rate,
            duration.as_secs_f64(),
            distances_rate,
            total_distances,
            average_recall
        );
    }

    Ok(())
}

fn print_help() {
    println!("aknn_benchmark usage:");
    println!("  --dims <n> (default=2)");
    println!("  --npoints <n> (default=100000)");
    println!("  --queries <n> (default=10000)");
    println!("  --seed <n> (default=42)");
    println!("  --csv <path>");
    println!("  --rates <csv> (default=0.01,0.02,0.05)");
}
