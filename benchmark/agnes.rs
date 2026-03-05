mod counting_distance;
mod data_loading;

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::atomic::Ordering;
use std::time::Instant;

use counting_distance::CountingEuclideanDistance;
use data_loading::read_numeric_data;
use hacs::hierarchical::{
    AverageLinkage, CentroidLinkage, CompleteLinkage, GroupAverageLinkage, MedianLinkage,
    MinimumVarianceLinkage, SingleLinkage, WardLinkage, WeightedAverageLinkage,
};
use hacs::{DataAccess, MatrixDataAccess, agnes, cut_dendrogram_by_number_of_clusters};

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);

    let csv_path = args
        .next()
        .ok_or_else(|| {
            "usage: cargo run --features benchmark --bin agnes_benchmark -- <csv_path> <k> <linkage>"
        })?;

    let k: usize = args
        .next()
        .ok_or_else(|| "missing k")?
        .parse()
        .map_err(|_| "k must be a positive integer")?;

    if k == 0 {
        return Err("k must be greater than 0".into());
    }

    let linkage_name = args.next().ok_or_else(|| "missing linkage type")?;

    // perform agnes using the selected linkage criterion; each variant has a
    // distinct type so we call the generic function inside the match to keep
    // the arms homogeneous.
    let rows = read_numeric_data(&csv_path)?;
    let n = rows.len();
    if n < 2 {
        return Err("CSV must contain at least two rows".into());
    }

    let distance = CountingEuclideanDistance::new();
    let distance_count = distance.counter();
    let data = MatrixDataAccess::with_distance(&rows, distance);

    // build condensed lower-triangular distance matrix
    let data_ref = &data;
    let condensed: Vec<_> = (1..n)
        .flat_map(|p| (0..p).map(move |q| data_ref.distance(p, q)))
        .collect();
    let distance_count_after_index = distance_count.load(Ordering::Relaxed);

    let start = Instant::now();
    let history = match linkage_name.as_str() {
        "single" => agnes(&condensed, n, SingleLinkage, false),
        "complete" => agnes(&condensed, n, CompleteLinkage, false),
        "average" => agnes(&condensed, n, AverageLinkage, false),
        "ward" => agnes(&condensed, n, WardLinkage, false),
        "centroid" => agnes(&condensed, n, CentroidLinkage, false),
        "median" => agnes(&condensed, n, MedianLinkage, false),
        "group_average" => agnes(&condensed, n, GroupAverageLinkage, false),
        "minimum_variance" => agnes(&condensed, n, MinimumVarianceLinkage, false),
        "weighted_average" => agnes(&condensed, n, WeightedAverageLinkage, false),
        _ => return Err("unknown linkage type".into()),
    };
    let elapsed = start.elapsed();
    let distance_count_after_algorithm = distance_count.load(Ordering::Relaxed);

    // extract flat clusters from the hierarchy with ELKI-style tie handling.
    let labels = cut_dendrogram_by_number_of_clusters(&history, k);
    let (cluster_sizes, _noise) = summarize_cluster_sizes(&labels);

    println!("time_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("cluster_count={k}");
    println!("noise_count=0");
    println!("cluster_sizes={}", format_cluster_sizes(&cluster_sizes));
    println!("distance_count_after_index={distance_count_after_index}");
    println!("distance_count_after_algorithm={distance_count_after_algorithm}");

    Ok(())
}

fn summarize_cluster_sizes(labels: &[usize]) -> (BTreeMap<usize, usize>, usize) {
    let cluster_sizes = labels.iter().fold(BTreeMap::new(), |mut m, &lbl| {
        *m.entry(lbl).or_insert(0) += 1;
        m
    });
    (cluster_sizes, 0)
}

fn format_cluster_sizes(cluster_sizes: &BTreeMap<usize, usize>) -> String {
    if cluster_sizes.is_empty() {
        return "none".to_string();
    }

    cluster_sizes
        .iter()
        .map(|(cluster_id, size)| format!("{cluster_id}:{size}"))
        .collect::<Vec<_>>()
        .join(",")
}
