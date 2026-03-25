mod counting_euclidean_distance;
mod counting_partial_distance;
mod data_loading;

use std::error::Error;
use std::sync::atomic::Ordering;
use std::time::Instant;

use counting_euclidean_distance::CountingEuclideanDistance;
use counting_partial_distance::CountingPartialDistance;
use data_loading::read_numeric_data;
use fuel::TableWithDistance;
use fuel::distance::EuclideanDistance;
use fuel::kd::{KdTree, MaxVarianceSplit};
use fuel::outlier::{k_nearest_neighbors_outlier as knn, outlier_detection_independence_neighbor};
use fuel::vptree::VPTree;
use rand::SeedableRng;
use rand::rngs::StdRng;

const RNG_SEED: u64 = 42;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeKind {
    Vp,
    Kd,
}

impl TreeKind {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "vp" | "vptree" => Ok(TreeKind::Vp),
            "kd" | "kdtree" => Ok(TreeKind::Kd),
            _ => Err("unknown tree kind, expected vp or kd".into()),
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);

    let csv_path = args
        .next()
        .ok_or("usage: cargo run --features benchmark --bin knn_benchmark -- <csv_path> <k> [--mode knn|odin] [--tree vp|kd]")?;

    let k: usize =
        args.next().ok_or("missing k")?.parse().map_err(|_| "k must be a positive integer")?;

    if k == 0 {
        return Err("k must be greater than 0".into());
    }

    // optional mode: "knn" (default) or "odin"; optional tree: "vp" (default) or "kd"
    let mut mode = "knn".to_string();
    let mut tree_kind = TreeKind::Vp;
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--mode=") {
            mode = value.to_string();
        } else if arg == "--mode" {
            mode = args.next().ok_or("--mode requires a value")?;
        } else if let Some(value) = arg.strip_prefix("--tree=") {
            tree_kind = TreeKind::parse(value)?;
        } else if arg == "--tree" {
            tree_kind = TreeKind::parse(&args.next().ok_or("--tree requires a value")?)?;
        } else {
            return Err(format!("unknown argument: {}", arg).into());
        }
    }

    let rows = read_numeric_data(&csv_path)?;
    if rows.len() < 2 {
        return Err("CSV must contain at least two rows".into());
    }

    // Build the data structure and count distance evaluations.
    let (tree_label, distance_count_after_index, scores, dist_count, elapsed) = match tree_kind {
        TreeKind::Vp => {
            let distance = CountingEuclideanDistance::new();
            let distance_count = distance.counter();
            let data = TableWithDistance::with_distance(&rows, distance);
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            let sample_size = rows.len();

            let start = Instant::now();
            let tree = VPTree::new(&data, sample_size, &mut rng);
            let distance_count_after_index = distance_count.load(Ordering::Relaxed);
            let scores = match mode.as_str() {
                "knn" => knn(&tree, &data, k),
                "odin" => outlier_detection_independence_neighbor(&tree, &data, k),
                other => return Err(format!("unknown mode: {}", other).into()),
            };
            let dist_count = distance_count.load(Ordering::Relaxed);
            let elapsed = start.elapsed();

            ("vp".to_string(), distance_count_after_index, scores, dist_count, elapsed)
        }
        TreeKind::Kd => {
            let kd_metric = CountingPartialDistance::new(EuclideanDistance);
            let data = TableWithDistance::with_distance(&rows, kd_metric.clone());
            let start = Instant::now();
            let tree = KdTree::new(&data, MaxVarianceSplit);
            let distance_count_after_index = kd_metric.count();
            let scores = match mode.as_str() {
                "knn" => knn(&tree, &data, k),
                "odin" => outlier_detection_independence_neighbor(&tree, &data, k),
                other => return Err(format!("unknown mode: {}", other).into()),
            };
            let dist_count = kd_metric.count();
            let elapsed = start.elapsed();

            ("kd".to_string(), distance_count_after_index, scores, dist_count, elapsed)
        }
    };

    let avg_score = scores.scores.iter().copied().sum::<f64>() / scores.scores.len() as f64;

    println!("tree={tree_label} time_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("avg_score={avg_score:.12}");
    println!("distance_count_after_index={distance_count_after_index}");
    println!("dist_count={dist_count}");

    Ok(())
}
