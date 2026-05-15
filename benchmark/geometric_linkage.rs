mod common;

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use common::{CountingDistance, read_numeric_data};
use fuel::cluster::hierarchical::extraction::cut_dendrogram_by_number_of_clusters;
use fuel::cluster::hierarchical::{
    CentroidLinkage, GroupAverageLinkage, MedianLinkage, MergeHistory, MinimumSumSquaresLinkage,
    MinimumVarianceIncreaseLinkage, MinimumVarianceLinkage, WardLinkage, agnes, anderberg,
    geometric_nn_chain, incremental_nn_chain, muellner, nn_chain, set_muellner,
};
use fuel::distance::SquaredEuclidean;
use fuel::search::vptree::VPTree;
use fuel::{CondensedDistanceMatrix, TableWithDistance};
use rand::SeedableRng;
use rand::rngs::StdRng;

const USAGE: &str = "usage: cargo run --features benchmark --bin geometric_linkage -- <csv_path> <cluster_count> <algorithm> <linkage>\n    algorithm: geometric_nn_chain|incremental_nn_chain|nn_chain|agnes|anderberg|muellner|set_muellner\n    linkage: average|centroid|median|ward|mivar|mnvar|missq|mnssq";

#[derive(Clone, Copy, Debug)]
enum Algorithm {
    GeometricNNChain,
    IncrementalNNChain,
    NNChain,
    Agnes,
    Anderberg,
    Muellner,
    SetMuellner,
}

#[derive(Clone, Copy, Debug)]
enum LinkageKind {
    GroupAverage,
    Centroid,
    Median,
    Ward,
    MinimumVarianceIncrease,
    MinimumVariance,
    MinimumSumSquares,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let csv_path = args.next().ok_or(USAGE)?;
    let cluster_count: usize = args
        .next()
        .ok_or(USAGE)?
        .parse()
        .map_err(|_| "cluster_count must be a positive integer")?;
    let algorithm_name = args.next().ok_or(USAGE)?;
    let linkage_name = args.next().ok_or(USAGE)?;

    let rows = read_numeric_data(&csv_path)?;
    let n = rows.len();
    if n < 2 {
        return Err("CSV must contain at least two rows".into());
    }
    if cluster_count == 0 || cluster_count > n {
        return Err("cluster_count must be between 1 and the number of rows".into());
    }

    let algorithm = parse_algorithm(&algorithm_name)?;
    let linkage = parse_linkage(&linkage_name)?;

    let distance = CountingDistance::new(SquaredEuclidean);
    let data: TableWithDistance<f64, Vec<f64>, CountingDistance<SquaredEuclidean>, f64> =
        TableWithDistance::with_distance(&rows, distance.clone());

    println!("dataset={csv_path}");
    println!("rows={n}");
    println!("dimensions={}", rows[0].len());

    println!("algorithm={algorithm_name}");
    println!("linkage={linkage_name}");
    println!("cluster_count={cluster_count}");
    let start = Instant::now();
    let history: MergeHistory<f64> = match algorithm {
        Algorithm::GeometricNNChain => match linkage {
            LinkageKind::GroupAverage => geometric_nn_chain(&data, GroupAverageLinkage),
            LinkageKind::Centroid => geometric_nn_chain(&data, CentroidLinkage),
            LinkageKind::Median => geometric_nn_chain(&data, MedianLinkage),
            LinkageKind::Ward => geometric_nn_chain(&data, WardLinkage),
            LinkageKind::MinimumVarianceIncrease => {
                geometric_nn_chain(&data, MinimumVarianceIncreaseLinkage)
            }
            LinkageKind::MinimumVariance => geometric_nn_chain(&data, MinimumVarianceLinkage),
            LinkageKind::MinimumSumSquares => geometric_nn_chain(&data, MinimumSumSquaresLinkage),
        },
        Algorithm::IncrementalNNChain => {
            let mut rng = StdRng::seed_from_u64(0);
            let tree = VPTree::new(&data, 10, &mut rng);
            match linkage {
                LinkageKind::GroupAverage => {
                    incremental_nn_chain(&tree, &data, GroupAverageLinkage)
                }
                LinkageKind::Centroid => incremental_nn_chain(&tree, &data, CentroidLinkage),
                LinkageKind::Median => incremental_nn_chain(&tree, &data, MedianLinkage),
                LinkageKind::Ward => incremental_nn_chain(&tree, &data, WardLinkage),
                LinkageKind::MinimumVarianceIncrease => {
                    incremental_nn_chain(&tree, &data, MinimumVarianceIncreaseLinkage)
                }
                LinkageKind::MinimumVariance => {
                    incremental_nn_chain(&tree, &data, MinimumVarianceLinkage)
                }
                LinkageKind::MinimumSumSquares => {
                    incremental_nn_chain(&tree, &data, MinimumSumSquaresLinkage)
                }
            }
        }
        Algorithm::NNChain => {
            let condensed = CondensedDistanceMatrix::new_from_data(&data);
            match linkage {
                LinkageKind::GroupAverage => nn_chain(&condensed, GroupAverageLinkage),
                LinkageKind::Centroid => nn_chain(&condensed, CentroidLinkage),
                LinkageKind::Median => nn_chain(&condensed, MedianLinkage),
                LinkageKind::Ward => nn_chain(&condensed, WardLinkage),
                LinkageKind::MinimumVarianceIncrease => {
                    nn_chain(&condensed, MinimumVarianceIncreaseLinkage)
                }
                LinkageKind::MinimumVariance => nn_chain(&condensed, MinimumVarianceLinkage),
                LinkageKind::MinimumSumSquares => nn_chain(&condensed, MinimumSumSquaresLinkage),
            }
        }
        Algorithm::Agnes => {
            let condensed = CondensedDistanceMatrix::new_from_data(&data);
            match linkage {
                LinkageKind::GroupAverage => agnes(&condensed, GroupAverageLinkage),
                LinkageKind::Centroid => agnes(&condensed, CentroidLinkage),
                LinkageKind::Median => agnes(&condensed, MedianLinkage),
                LinkageKind::Ward => agnes(&condensed, WardLinkage),
                LinkageKind::MinimumVarianceIncrease => {
                    agnes(&condensed, MinimumVarianceIncreaseLinkage)
                }
                LinkageKind::MinimumVariance => agnes(&condensed, MinimumVarianceLinkage),
                LinkageKind::MinimumSumSquares => agnes(&condensed, MinimumSumSquaresLinkage),
            }
        }
        Algorithm::Anderberg => {
            let condensed = CondensedDistanceMatrix::new_from_data(&data);
            match linkage {
                LinkageKind::GroupAverage => anderberg(&condensed, GroupAverageLinkage),
                LinkageKind::Centroid => anderberg(&condensed, CentroidLinkage),
                LinkageKind::Median => anderberg(&condensed, MedianLinkage),
                LinkageKind::Ward => anderberg(&condensed, WardLinkage),
                LinkageKind::MinimumVarianceIncrease => {
                    anderberg(&condensed, MinimumVarianceIncreaseLinkage)
                }
                LinkageKind::MinimumVariance => anderberg(&condensed, MinimumVarianceLinkage),
                LinkageKind::MinimumSumSquares => anderberg(&condensed, MinimumSumSquaresLinkage),
            }
        }
        Algorithm::Muellner => {
            let condensed = CondensedDistanceMatrix::new_from_data(&data);
            match linkage {
                LinkageKind::GroupAverage => muellner(&condensed, GroupAverageLinkage),
                LinkageKind::Centroid => muellner(&condensed, CentroidLinkage),
                LinkageKind::Median => muellner(&condensed, MedianLinkage),
                LinkageKind::Ward => muellner(&condensed, WardLinkage),
                LinkageKind::MinimumVarianceIncrease => {
                    muellner(&condensed, MinimumVarianceIncreaseLinkage)
                }
                LinkageKind::MinimumVariance => muellner(&condensed, MinimumVarianceLinkage),
                LinkageKind::MinimumSumSquares => muellner(&condensed, MinimumSumSquaresLinkage),
            }
        }
        Algorithm::SetMuellner => match linkage {
            LinkageKind::GroupAverage => set_muellner::<_, GroupAverageLinkage, _, _>(&data),
            LinkageKind::Ward => set_muellner::<_, WardLinkage, _, _>(&data),
            LinkageKind::MinimumVarianceIncrease => {
                set_muellner::<_, MinimumVarianceIncreaseLinkage, _, _>(&data)
            }
            LinkageKind::MinimumVariance => set_muellner::<_, MinimumVarianceLinkage, _, _>(&data),
            LinkageKind::MinimumSumSquares => {
                set_muellner::<_, MinimumSumSquaresLinkage, _, _>(&data)
            }
            _ => panic!("set_muellner does not support the chosen linkage"),
        },
    }?;
    let elapsed = start.elapsed();

    let labels = cut_dendrogram_by_number_of_clusters(&history, cluster_count);
    let (cluster_sizes, _) = summarize_cluster_sizes(&labels);
    let merge_height_sum: f64 = history.iter().map(|m| m.distance).sum();
    let last_merge_height = history.iter().last().map(|m| m.distance).unwrap_or(0.0);

    println!("merge_height_sum={merge_height_sum:.15}");
    println!("last_merge_height={last_merge_height:.15}");
    println!("cluster_sizes={}", format_cluster_sizes(&cluster_sizes));
    println!("distance_count={}", distance.count());
    println!("time_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);

    Ok(())
}

fn parse_algorithm(name: &str) -> Result<Algorithm, Box<dyn Error>> {
    match name.to_lowercase().as_str() {
        "geometric_nn_chain" | "geometric-nn-chain" | "gnnc" => Ok(Algorithm::GeometricNNChain),
        "incremental_nn_chain" | "incremental-nn-chain" | "innc" => {
            Ok(Algorithm::IncrementalNNChain)
        }
        "nn_chain" | "nn-chain" | "nnc" => Ok(Algorithm::NNChain),
        "agnes" => Ok(Algorithm::Agnes),
        "anderberg" => Ok(Algorithm::Anderberg),
        "muellner" => Ok(Algorithm::Muellner),
        "set_muellner" | "set-muellner" | "setmuellner" => Ok(Algorithm::SetMuellner),
        _ => Err(format!("unknown algorithm: {name}").into()),
    }
}

fn parse_linkage(name: &str) -> Result<LinkageKind, Box<dyn Error>> {
    match name.to_lowercase().as_str() {
        "average" | "group_average" | "group-average" => Ok(LinkageKind::GroupAverage),
        "centroid" => Ok(LinkageKind::Centroid),
        "median" => Ok(LinkageKind::Median),
        "ward" => Ok(LinkageKind::Ward),
        "mivar" | "minimum_variance_increase" | "minimum-variance-increase" => {
            Ok(LinkageKind::MinimumVarianceIncrease)
        }
        "mnvar" | "minimum_variance" | "minimum-variance" => Ok(LinkageKind::MinimumVariance),
        "missq" | "mnssq" | "minimum_sum_squares" | "minimum-sum-squares" => {
            Ok(LinkageKind::MinimumSumSquares)
        }
        _ => Err(format!("unknown linkage: {name}").into()),
    }
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
