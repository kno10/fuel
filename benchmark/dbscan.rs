mod common;

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use common::{CountingDistance, read_numeric_data};
use fuel::TableWithDistance;
use fuel::cluster::dbscan::{NOISE, dbscan};
use fuel::cluster::parallel_dbscan::parallel_dbscan;
use fuel::distance::Euclidean;
use fuel::search::kdtree::{KdTree, MaxVarianceSplit};
use fuel::search::vptree::VPTree;
use rand::SeedableRng;
use rand::rngs::StdRng;

const RNG_SEED: u64 = 42;
const USAGE: &str = "usage: cargo run --features benchmark --bin dbscan -- <csv_path> <eps> <min_points> [--mode=<sequential|parallel|both>] [--tree=<vp|kd>]";

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
    let (csv_path, eps, min_points, mode, tree_kind) = parse_cli_args(std::env::args().skip(1))?;

    let rows = read_numeric_data(&csv_path)?;
    if rows.len() < 2 {
        return Err("CSV must contain at least two rows".into());
    }

    let mut results = Vec::new();
    if mode.should_run(Variant::Sequential) {
        results.push(benchmark_variant(&rows, eps, min_points, Variant::Sequential, tree_kind));
    }
    if mode.should_run(Variant::Parallel) {
        results.push(benchmark_variant(&rows, eps, min_points, Variant::Parallel, tree_kind));
    }

    for result in results {
        print_result(&result);
    }

    Ok(())
}

fn parse_cli_args<I>(args: I) -> Result<(String, f64, usize, Mode, TreeKind), Box<dyn Error>>
where
    I: IntoIterator<Item = String>,
{
    let mut mode = Mode::Both;
    let mut tree_kind = TreeKind::Vp;
    let mut positional = Vec::new();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--mode=") {
            mode = Mode::parse(value)?;
        } else if arg == "--mode" {
            let value = iter.next().ok_or_else(|| "--mode requires a value".to_string())?;
            mode = Mode::parse(&value)?;
        } else if let Some(value) = arg.strip_prefix("--tree=") {
            tree_kind = TreeKind::parse(value)?;
        } else if arg == "--tree" {
            let value = iter.next().ok_or_else(|| "--tree requires a value".to_string())?;
            tree_kind = TreeKind::parse(&value)?;
        } else {
            positional.push(arg);
        }
    }

    if positional.len() != 3 {
        return Err(USAGE.into());
    }

    let mut pos_iter = positional.into_iter();
    let csv_path = pos_iter.next().unwrap();
    let eps_str = pos_iter.next().unwrap();
    let min_points_str = pos_iter.next().unwrap();

    let eps: f64 = eps_str.parse().map_err(|_| "eps must be a non-negative number".to_string())?;
    let min_points: usize =
        min_points_str.parse().map_err(|_| "min_points must be a positive integer".to_string())?;

    if eps < 0.0 {
        return Err("eps must be non-negative".into());
    }

    if min_points == 0 {
        return Err("min_points must be greater than 0".into());
    }

    Ok((csv_path, eps, min_points, mode, tree_kind))
}

fn benchmark_variant(
    rows: &[Vec<f64>], eps: f64, min_points: usize, variant: Variant, tree_kind: TreeKind,
) -> BenchmarkResult {
    let (tree_label, distance_after_index, labels, distance_after_algorithm, elapsed) =
        match tree_kind {
            TreeKind::Vp => {
                let distance = CountingDistance::new(Euclidean);
                let data = TableWithDistance::with_distance(rows, distance.clone());
                let mut rng = StdRng::seed_from_u64(RNG_SEED);

                let start = Instant::now();
                let tree = VPTree::new(&data, rows.len(), &mut rng);
                let distance_after_index = distance.count() as u64;
                let labels = match variant {
                    Variant::Sequential => dbscan(&tree, &data, eps, min_points),
                    Variant::Parallel => parallel_dbscan(&tree, &data, eps, min_points),
                };
                let distance_after_algorithm = distance.count() as u64;
                let elapsed = start.elapsed();
                ("vp".to_string(), distance_after_index, labels, distance_after_algorithm, elapsed)
            }
            TreeKind::Kd => {
                let kd_metric = CountingDistance::new(Euclidean);
                let data = TableWithDistance::with_distance(rows, kd_metric.clone());
                let start = Instant::now();
                let tree = KdTree::new(&data, MaxVarianceSplit);
                let distance_after_index = kd_metric.count() as u64;
                let labels = match variant {
                    Variant::Sequential => dbscan(&tree, &data, eps, min_points),
                    Variant::Parallel => parallel_dbscan(&tree, &data, eps, min_points),
                };
                let distance_after_algorithm = kd_metric.count() as u64;
                let elapsed = start.elapsed();
                ("kd".to_string(), distance_after_index, labels, distance_after_algorithm, elapsed)
            }
        };

    let (cluster_sizes, noise_count) = summarize_cluster_sizes(&labels);
    BenchmarkResult {
        variant,
        tree_label,
        time_ms: elapsed.as_secs_f64() * 1_000.0,
        cluster_count: cluster_sizes.len(),
        noise_count,
        cluster_sizes: format_cluster_sizes(&cluster_sizes),
        distance_count_after_index: distance_after_index,
        dist_count: distance_after_algorithm,
    }
}

fn print_result(result: &BenchmarkResult) {
    println!(
        "variant={} tree={} time_ms={:.3} cluster_count={} noise_count={} cluster_sizes={} distance_count_after_index={} dist_count= {}",
        result.variant.label(),
        result.tree_label,
        result.time_ms,
        result.cluster_count,
        result.noise_count,
        result.cluster_sizes,
        result.distance_count_after_index,
        result.dist_count
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Sequential,
    Parallel,
}

impl Variant {
    fn label(&self) -> &'static str {
        match self {
            Variant::Sequential => "sequential",
            Variant::Parallel => "parallel",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Sequential,
    Parallel,
    Both,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "sequential" => Ok(Mode::Sequential),
            "parallel" => Ok(Mode::Parallel),
            "both" => Ok(Mode::Both),
            _ => Err("mode must be sequential, parallel, or both".into()),
        }
    }

    fn should_run(self, variant: Variant) -> bool {
        matches!(self, Mode::Both)
            || (self == Mode::Sequential && variant == Variant::Sequential)
            || (self == Mode::Parallel && variant == Variant::Parallel)
    }
}

struct BenchmarkResult {
    variant: Variant,
    tree_label: String,
    time_ms: f64,
    cluster_count: usize,
    noise_count: usize,
    cluster_sizes: String,
    distance_count_after_index: u64,
    dist_count: u64,
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
