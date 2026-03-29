mod common;

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use common::{CountingDistance, read_numeric_data};
use fuel::cluster::hdbscan::extraction::{ExtractedHierarchy, extract_simplified_hierarchy};
use fuel::cluster::hierarchical::{
    AverageLinkage, CentroidLinkage, CompleteLinkage, GroupAverageLinkage, MedianLinkage, Merge,
    MinimumVarianceLinkage, SingleLinkage, WardLinkage, WeightedAverageLinkage, agnes,
};
use fuel::distance::Euclidean;
use fuel::{DistanceData, TableWithDistance};

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);

    let csv_path = args.next().ok_or(
        "usage: cargo run --features benchmark --bin agnes_benchmark -- <csv_path> <k> <linkage>",
    )?;

    let k: usize =
        args.next().ok_or("missing k")?.parse().map_err(|_| "k must be a positive integer")?;

    if k == 0 {
        return Err("k must be greater than 0".into());
    }

    let linkage_name = args.next().ok_or("missing linkage type")?;

    // perform agnes using the selected linkage criterion; each variant has a
    // distinct type so we call the generic function inside the match to keep
    // the arms homogeneous.
    let rows = read_numeric_data(&csv_path)?;
    let n = rows.len();
    if n < 2 {
        return Err("CSV must contain at least two rows".into());
    }

    let distance = CountingDistance::new(Euclidean);
    let data: TableWithDistance<f64, Vec<f64>, CountingDistance<Euclidean>, f64> =
        TableWithDistance::with_distance(&rows, distance.clone());

    // build condensed lower-triangular distance matrix
    let data_ref = &data;
    let condensed: Vec<_> =
        (1..n).flat_map(|p| (0..p).map(move |q| data_ref.distance(p, q))).collect();
    let distance_count_after_index = distance.count();

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
    let distance_count_after_algorithm = distance.count();

    // extract flat clusters from the hierarchy with ELKI-style tie handling.
    let labels = labels_from_simplified_hierarchy(&history, n, k);
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

// same helper functions as in `single_link.rs`

fn collect_subtree_members(node: usize, extracted: &ExtractedHierarchy<f64>, out: &mut Vec<usize>) {
    out.extend(extracted.nodes[node].members.iter().copied());
    for &child in &extracted.nodes[node].children {
        collect_subtree_members(child, extracted, out);
    }
}

fn labels_from_frontier(
    extracted: &ExtractedHierarchy<f64>, frontier: &[usize], n: usize,
) -> Vec<usize> {
    let mut labels = vec![0; n];

    for (cid, &node) in frontier.iter().enumerate() {
        let mut members = Vec::new();
        collect_subtree_members(node, extracted, &mut members);
        for point in members {
            if point < n && labels[point] == 0 {
                labels[point] = cid;
            }
        }
    }
    labels
}

fn labels_from_simplified_hierarchy(
    history: &[Merge<f64>], n: usize, min_clusters: usize,
) -> Vec<usize> {
    let extracted = extract_simplified_hierarchy(history, None, 1);
    assert!(min_clusters > 0, "min_clusters must be positive");
    if extracted.roots.is_empty() {
        return vec![0; n];
    }

    let mut frontier = extracted.roots.clone();
    while frontier.len() < min_clusters {
        let mut best_pos = None;
        let mut best_dist = f64::NEG_INFINITY;
        for (i, &node) in frontier.iter().enumerate() {
            if extracted.nodes[node].children.is_empty() {
                continue;
            }
            let d = extracted.nodes[node].distance;
            if d > best_dist {
                best_dist = d;
                best_pos = Some(i);
            }
        }

        let Some(pos) = best_pos else {
            break;
        };
        let node = frontier.swap_remove(pos);
        frontier.extend(&extracted.nodes[node].children);
    }

    labels_from_frontier(&extracted, &frontier, n)
}
