mod common;

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use common::{CountingDistance, read_numeric_data};
use fuel::cluster::dbscan::NOISE;
use fuel::cluster::hdbscan::extraction::extract_clusters_with_noise;
use fuel::cluster::hdbscan::{
    HdbscanHierarchy, boruvka_searchers_hdbscan, buffered_search_hdbscan, hdbscan_prim,
    heap_of_searchers_hdbscan, lazy_buffered_search_hdbscan, restarting_search_hdbscan,
    slink_hdbscan,
};
use fuel::cluster::hierarchical::MergeHistory;
use fuel::distance::Euclidean;
use fuel::search::vptree::VPTree;
use fuel::{Data, TableWithDistance};
use rand::SeedableRng;
use rand::rngs::StdRng;

const RNG_SEED: u64 = 42;
const BUFFERED_SLACK: usize = 4;
const USAGE: &str = "usage: cargo run --features benchmark --bin hdbscan -- <csv_path> <min_points> [--algorithms=<heap|boruvka|restarting|buffered|lazy-buffered|linear|slink|all>]";

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let (csv_path, min_points, selection) = parse_cli_args(std::env::args().skip(1))?;

    let rows = read_numeric_data(&csv_path)?;
    if rows.is_empty() {
        return Err("CSV must contain at least one row".into());
    }

    // build index (VPTree) once if any variant needs it
    let any_tree = selection.variants().iter().any(|v| v.requires_tree());
    let mut maybe_tree: Option<VPTree<f64>> = None;

    if any_tree {
        let distance = CountingDistance::new(Euclidean);
        let data = TableWithDistance::with_distance(&rows, distance.clone());
        let mut rng = StdRng::seed_from_u64(RNG_SEED);

        let start_idx = Instant::now();
        let tree = VPTree::new(&data, rows.len(), &mut rng);
        let index_time_ms = start_idx.elapsed().as_secs_f64() * 1_000.0;
        let index_dist_count = distance.count();
        maybe_tree = Some(tree);

        // print index statistics once at the beginning
        println!("index time_ms={:.3} distance_count={}", index_time_ms, index_dist_count);
    }

    // prepare data access for algorithm runs (new counter)
    let distance = CountingDistance::new(Euclidean);
    let data: TableWithDistance<f64, Vec<f64>, CountingDistance<Euclidean>, f64> =
        TableWithDistance::with_distance(&rows, distance.clone());

    for &variant in selection.variants() {
        let result = benchmark_variant(&data, min_points, variant, maybe_tree.as_ref(), &distance);
        print_result(&result);
    }

    Ok(())
}

fn parse_cli_args<I>(args: I) -> Result<(String, usize, VariantSelection), Box<dyn Error>>
where
    I: IntoIterator<Item = String>,
{
    let mut selection = VariantSelection::all();
    let mut positional = Vec::new();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        if let Some(value) = arg.strip_prefix("--algorithms=") {
            selection = VariantSelection::from_list(value)?;
        } else if arg == "--algorithms" {
            let value = iter.next().ok_or_else(|| "--algorithms requires a value".to_string())?;
            selection = VariantSelection::from_list(&value)?;
        } else if let Some(value) = arg.strip_prefix("--variant=") {
            // legacy alias for compatibility with older benchmark flag naming
            selection = VariantSelection::from_list(value)?;
        } else if arg == "--variant" {
            let value = iter.next().ok_or_else(|| "--variant requires a value".to_string())?;
            selection = VariantSelection::from_list(&value)?;
        } else {
            positional.push(arg);
        }
    }

    if positional.len() != 2 {
        return Err(USAGE.into());
    }

    let mut pos_iter = positional.into_iter();
    let csv_path = pos_iter.next().unwrap();
    let min_points_str = pos_iter.next().unwrap();

    let min_points: usize =
        min_points_str.parse().map_err(|_| "min_points must be a positive integer".to_string())?;

    if min_points == 0 {
        return Err("min_points must be greater than 0".into());
    }

    Ok((csv_path, min_points, selection))
}

fn benchmark_variant(
    data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<Euclidean>, f64>,
    min_points: usize, variant: Variant, prebuilt_tree: Option<&VPTree<f64>>,
    distance: &CountingDistance<Euclidean>,
) -> BenchmarkResult {
    let baseline = distance.count();
    let start = Instant::now();
    let hierarchy = variant.run(prebuilt_tree, data, min_points);
    let after = distance.count();
    let dist_count = after.saturating_sub(baseline) as u64;
    let elapsed = start.elapsed();

    let mst_weight: f64 = hierarchy.merges.iter().map(|m| m.distance).sum();

    let labels = extract_labels(&hierarchy.merges, data.len(), min_points);
    let (cluster_sizes, noise_count) = summarize_cluster_sizes(&labels);

    BenchmarkResult {
        variant,
        time_ms: elapsed.as_secs_f64() * 1_000.0,
        mst_weight,
        cluster_count: cluster_sizes.len(),
        noise_count,
        cluster_sizes: format_cluster_sizes(&cluster_sizes),
        dist_count,
    }
}

fn extract_labels(history: &MergeHistory<f64>, n: usize, min_points: usize) -> Vec<isize> {
    let num_clusters = (n / min_points).max(1);
    extract_clusters_with_noise(history, num_clusters, min_points)
}

fn print_result(result: &BenchmarkResult) {
    println!(
        "variant={} time_ms={:.3} mst_weight={:.15} cluster_count={} noise_count={} cluster_sizes={} dist_count={}",
        result.variant.label(),
        result.time_ms,
        result.mst_weight,
        result.cluster_count,
        result.noise_count,
        result.cluster_sizes,
        result.dist_count
    );
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    Prim,
    Slink,
    Heap,
    Restarting,
    Buffered,
    LazyBuffered,
    Boruvka,
}

impl Variant {
    const ALL: [Variant; 5] =
        [Self::Heap, Self::Boruvka, Self::Restarting, Self::Slink, Self::Prim];

    fn label(&self) -> &'static str {
        match self {
            Variant::Heap => "heap_of_searchers",
            Variant::Boruvka => "boruvka_searchers",
            Variant::Restarting => "restarting_search",
            Variant::Prim => "prim",
            Variant::Slink => "slink",
            Variant::Buffered => "buffered_search",
            Variant::LazyBuffered => "lazy_buffered_search",
        }
    }

    fn requires_tree(&self) -> bool { !matches!(self, Variant::Prim | Variant::Slink) }

    fn run(
        &self, tree: Option<&VPTree<f64>>,
        data: &TableWithDistance<'_, f64, Vec<f64>, CountingDistance<Euclidean>, f64>,
        min_points: usize,
    ) -> HdbscanHierarchy<f64> {
        match self {
            Variant::Heap => {
                heap_of_searchers_hdbscan(tree.expect("tree required"), data, min_points)
            }
            Variant::Boruvka => {
                boruvka_searchers_hdbscan(tree.expect("tree required"), data, min_points)
            }
            Variant::Restarting => {
                restarting_search_hdbscan(tree.expect("tree required"), data, min_points)
            }
            Variant::Prim => hdbscan_prim(data, min_points),
            Variant::Slink => slink_hdbscan(data, min_points),
            Variant::Buffered => buffered_search_hdbscan(
                tree.expect("tree required"),
                data,
                min_points,
                BUFFERED_SLACK,
            ),
            Variant::LazyBuffered => {
                lazy_buffered_search_hdbscan(tree.expect("tree required"), data, min_points, 1)
            }
        }
    }

    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value {
            "heap" => Ok(Variant::Heap),
            "boruvka" => Ok(Variant::Boruvka),
            "restarting" => Ok(Variant::Restarting),
            "buffered" => Ok(Variant::Buffered),
            "lazy-buffered" | "lazy_buffered" | "lbssl" => Ok(Variant::LazyBuffered),
            "linear" => Ok(Variant::Prim),
            "slink" => Ok(Variant::Slink),
            _ => Err(
                "unknown variant, expected heap, boruvka, restarting, buffered, lazy-buffered, linear, or slink"
                    .into(),
            ),
        }
    }
}

struct VariantSelection {
    variants: Vec<Variant>,
}

impl VariantSelection {
    fn all() -> Self { Self { variants: Variant::ALL.to_vec() } }

    fn variants(&self) -> &[Variant] { &self.variants }

    fn from_list(value: &str) -> Result<Self, Box<dyn Error>> {
        if value == "all" {
            return Ok(Self::all());
        }
        let mut variants = Vec::new();
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let variant = Variant::parse(token)?;
            if !variants.contains(&variant) {
                variants.push(variant);
            }
        }
        if variants.is_empty() {
            return Err("variant list must contain at least one entry".into());
        }
        Ok(Self { variants })
    }
}

struct BenchmarkResult {
    variant: Variant,
    time_ms: f64,
    mst_weight: f64,
    cluster_count: usize,
    noise_count: usize,
    cluster_sizes: String,
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
