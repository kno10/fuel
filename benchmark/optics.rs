mod counting_euclidean_distance;
mod counting_partial_distance;
mod data_loading;

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::atomic::Ordering;
use std::time::Instant;

use counting_euclidean_distance::CountingEuclideanDistance;
use counting_partial_distance::CountingPartialDistance;
use data_loading::read_numeric_data;
use fuel::TableWithDistance;
use fuel::cluster::dbscan::NOISE;
use fuel::cluster::optics::{extract_xi_labels, optics};
use fuel::distance::EuclideanDistance;
use fuel::kd::{KdTree, MaxVarianceSplit};
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

    let csv_path = args.next().ok_or(
        "usage: cargo run --features benchmark --bin optics_benchmark -- <csv_path> <max_eps> <min_points> <xi> [--tree vp|kd]",
    )?;

    let max_eps: f64 = args
        .next()
        .ok_or("missing max_eps")?
        .parse()
        .map_err(|_| "max_eps must be a non-negative number")?;

    let min_points: usize = args
        .next()
        .ok_or("missing min_points")?
        .parse()
        .map_err(|_| "min_points must be a positive integer")?;

    let xi: f64 =
        args.next().ok_or("missing xi")?.parse().map_err(|_| "xi must be a number in (0, 1)")?;

    let mut tree_kind = TreeKind::Vp;
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--tree=") {
            tree_kind = TreeKind::parse(value)?;
        } else if arg == "--tree" {
            tree_kind = TreeKind::parse(&args.next().ok_or("--tree requires a value")?)?;
        } else {
            return Err(format!("unknown argument: {}", arg).into());
        }
    }

    if max_eps < 0.0 {
        return Err("max_eps must be non-negative".into());
    }

    if min_points == 0 {
        return Err("min_points must be greater than 0".into());
    }

    if !(0.0..1.0).contains(&xi) {
        return Err("xi must be in (0, 1)".into());
    }

    let rows = read_numeric_data(&csv_path)?;
    if rows.len() < 2 {
        return Err("CSV must contain at least two rows".into());
    }

    let (tree_label, distance_count_after_index, result, dist_count, elapsed) = match tree_kind {
        TreeKind::Vp => {
            let distance = CountingEuclideanDistance::new();
            let distance_count = distance.counter();
            let data = TableWithDistance::with_distance(&rows, distance);
            let mut rng = StdRng::seed_from_u64(RNG_SEED);
            let sample_size = rows.len();

            let start = Instant::now();
            let tree = VPTree::new(&data, sample_size, &mut rng);
            let distance_count_after_index = distance_count.load(Ordering::Relaxed);
            let result = optics(&tree, &data, max_eps, min_points);
            let dist_count = distance_count.load(Ordering::Relaxed);
            let elapsed = start.elapsed();

            ("vp".to_string(), distance_count_after_index, result, dist_count, elapsed)
        }
        TreeKind::Kd => {
            let kd_metric = CountingPartialDistance::new(EuclideanDistance);
            let data = TableWithDistance::with_distance(&rows, kd_metric.clone());
            let start = Instant::now();
            let tree = KdTree::new(&data, MaxVarianceSplit);
            let distance_count_after_index = kd_metric.count();
            let result = optics(&tree, &data, max_eps, min_points);
            let dist_count = kd_metric.count();
            let elapsed = start.elapsed();

            ("kd".to_string(), distance_count_after_index, result, dist_count, elapsed)
        }
    };

    let labels = extract_xi_labels(&result, xi, min_points);

    let (cluster_sizes, noise_count) = summarize_cluster_sizes(&labels);

    println!("tree={tree_label} time_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("cluster_count={}", cluster_sizes.len());
    println!("noise_count={noise_count}");
    println!("cluster_sizes={}", format_cluster_sizes(&cluster_sizes));
    println!("distance_count_after_index={distance_count_after_index}");
    println!("dist_count={dist_count}");

    Ok(())
}

fn summarize_cluster_sizes(labels: &[isize]) -> (BTreeMap<isize, usize>, usize) {
    let mut cluster_sizes = BTreeMap::new();
    let mut noise_count = 0usize;

    for &label in labels {
        if label == NOISE {
            noise_count += 1;
        } else {
            *cluster_sizes.entry(label).or_insert(0) += 1;
        }
    }

    (cluster_sizes, noise_count)
}

fn format_cluster_sizes(cluster_sizes: &BTreeMap<isize, usize>) -> String {
    if cluster_sizes.is_empty() {
        return "none".to_string();
    }

    cluster_sizes
        .iter()
        .map(|(cluster_id, size)| format!("{cluster_id}:{size}"))
        .collect::<Vec<_>>()
        .join(",")
}
