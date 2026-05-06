mod common;

use std::collections::BTreeMap;
use std::error::Error;
use std::time::Instant;

use common::{CountingDistance, read_numeric_data_with_limit};
use fuel::TableWithDistance;
use fuel::cluster::hdbscan::extraction::{ExtractedHierarchy, extract_simplified_hierarchy};
use fuel::cluster::hierarchical::{
    MergeHistory, SingleLinkage, agnes, anderberg, boruvka_searchers_single_link,
    buffered_search_single_link, heap_of_searchers_single_link, lazy_buffered_search_single_link,
    muellner, nn_chain, restarting_search_single_link, slink,
};
use fuel::condensed_distance_matrix::CondensedDistanceMatrix;
use fuel::distance::Euclidean;
use fuel::search::kdtree::{KdTree, MaxVarianceSplit};
use fuel::search::vptree::VPTree;
use rand::SeedableRng;
use rand::rngs::StdRng;

const DEFAULT_BUFFERED_SLACK: usize = 4;
const DEFAULT_TREE_SAMPLE: usize = 16;
const DEFAULT_VPTREE_SEED: u64 = 0xDEAD_BEEF;
const DEFAULT_CLUSTER_COUNT: usize = 10;
const USAGE: &str = "usage: cargo run --features benchmark --bin single_link -- <csv_path> <n> [--algorithms LIST] [--tree vp|kd] [--tree-sample SIZE] [--buffered-slack SIZE] [--cluster-count K] [--seed SEED]\n    LIST is comma-separated names: boruvka,heap,restart,buffered,lazy-buffered,slink,agnes,anderberg,muellner,nnchain (default all except agnes)";

#[derive(Clone, Copy, Debug)]
enum TreeKind {
    Vp,
    Kd,
}

impl TreeKind {
    fn parse(arg: &str) -> Result<Self, String> {
        match arg.to_lowercase().as_str() {
            "vp" | "vptree" => Ok(TreeKind::Vp),
            "kd" | "kdtree" => Ok(TreeKind::Kd),
            _ => Err(format!("unknown tree kind: {arg}")),
        }
    }
}

struct Config {
    csv_path: String,
    requested_rows: usize,
    tree_kind: TreeKind,
    tree_sample_size: usize,
    buffered_slack: usize,
    cluster_count: usize,
    seed: u64,
    alg_arg: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;

    let mut rows = read_numeric_data_with_limit(&config.csv_path, Some(config.requested_rows))?;
    if rows.len() < config.requested_rows {
        let rows_len = rows.len();
        return Err(format!(
            "CSV {csv_path} only contains {rows_len} rows but {requested_rows} were requested",
            csv_path = config.csv_path,
            rows_len = rows_len,
            requested_rows = config.requested_rows,
        )
        .into());
    }
    rows.truncate(config.requested_rows);
    let used_rows = rows.len();

    let dimension = rows.first().map_or(0, Vec::len);
    let distance = CountingDistance::new(Euclidean);
    let data: TableWithDistance<f64, Vec<f64>, CountingDistance<Euclidean>, f64> =
        TableWithDistance::with_distance(&rows, distance.clone());
    let sample_size = config.tree_sample_size.min(used_rows).max(1);

    print_run_information(&config, used_rows, dimension, sample_size);

    let algorithms = build_algorithm_list(config.alg_arg.as_deref(), config.buffered_slack)?;
    for algorithm in algorithms {
        let label = algorithm.label();
        let baseline = distance.count();
        let start = Instant::now();
        let history = run_single_link_algorithm(
            algorithm,
            config.tree_kind,
            config.seed,
            sample_size,
            &data,
        );
        let elapsed = start.elapsed();
        let after = distance.count();
        let dist_count = after.saturating_sub(baseline);
        let mst_weight: f64 = history.iter().map(|m| m.distance).sum();
        let labels = labels_from_simplified_hierarchy(&history, used_rows, config.cluster_count);
        let (cluster_sizes, _noise) = summarize_cluster_sizes(&labels);

        println!(
            "algorithm={label}, time_ms={:.3} mst_weight={:.15} cluster_count={} noise_count=0 cluster_sizes={} dist_count={}",
            elapsed.as_secs_f64() * 1_000.0,
            mst_weight,
            cluster_sizes.len(),
            format_cluster_sizes(&cluster_sizes),
            dist_count
        );
    }

    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut args = std::env::args();
    args.next();
    let csv_path = args.next().ok_or_else(|| usage_error())?;
    let requested_rows = args
        .next()
        .ok_or_else(|| usage_error())?
        .parse::<usize>()
        .map_err(|_| "data size must be a positive integer")?;
    if requested_rows < 2 {
        return Err("data size must be at least 2".into());
    }

    let mut tree_kind = TreeKind::Vp;
    let mut tree_sample_size = DEFAULT_TREE_SAMPLE;
    let mut buffered_slack = DEFAULT_BUFFERED_SLACK;
    let mut cluster_count = DEFAULT_CLUSTER_COUNT;
    let mut seed = DEFAULT_VPTREE_SEED;
    let mut alg_arg: Option<String> = None;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--algorithms" => alg_arg = Some(args.next().ok_or_else(|| usage_error())?),
            "--tree" => {
                tree_kind = TreeKind::parse(&args.next().ok_or_else(|| usage_error())?)?;
            }
            "--tree-sample" => tree_sample_size = parse_positive_usize(&mut args, &flag)?,
            "--buffered-slack" => buffered_slack = parse_positive_usize(&mut args, &flag)?,
            "--cluster-count" => cluster_count = parse_positive_usize(&mut args, &flag)?,
            "--seed" => seed = parse_seed(&mut args, &flag)?,
            _ => {
                return Err(Box::<dyn Error>::from(format!("unexpected argument '{flag}'")));
            }
        }
    }

    Ok(Config {
        csv_path,
        requested_rows,
        tree_kind,
        tree_sample_size,
        buffered_slack,
        cluster_count,
        seed,
        alg_arg,
    })
}

fn print_run_information(config: &Config, used_rows: usize, dimension: usize, sample_size: usize) {
    println!("dataset={}", config.csv_path);
    println!("data_rows={used_rows}");
    println!("dimensions={dimension}");
    println!("tree_kind={:?}", config.tree_kind);
    println!("tree_sample_size={sample_size}");
    println!("buffered_slack={}", config.buffered_slack);
    println!("seed={}", config.seed);
}

fn build_algorithm_list(
    alg_arg: Option<&str>,
    buffered_slack: usize,
) -> Result<Vec<SingleLinkAlgorithm>, Box<dyn Error>> {
    let all = vec![
        SingleLinkAlgorithm::Boruvka,
        SingleLinkAlgorithm::HeapOfSearchers,
        SingleLinkAlgorithm::RestartingSearch,
        SingleLinkAlgorithm::Slink,
        SingleLinkAlgorithm::Agnes,
        SingleLinkAlgorithm::Anderberg,
        SingleLinkAlgorithm::Muellner,
        SingleLinkAlgorithm::NNChain,
    ];

    if let Some(s) = alg_arg {
        let mut out = Vec::new();
        for part in s.split(',') {
            match part.trim().to_lowercase().as_str() {
                "boruvka" => out.push(SingleLinkAlgorithm::Boruvka),
                "heap" | "hssl" | "heap_of_searchers" => {
                    out.push(SingleLinkAlgorithm::HeapOfSearchers);
                }
                "restart" | "restarting" | "rssl" => {
                    out.push(SingleLinkAlgorithm::RestartingSearch);
                }
                "buffered" => {
                    out.push(SingleLinkAlgorithm::BufferedSearch { slack: buffered_slack });
                }
                "lazy-buffered" | "lazy_buffered" | "lbssl" => {
                    out.push(SingleLinkAlgorithm::LazyBufferedSearch { slack: buffered_slack });
                }
                "slink" => out.push(SingleLinkAlgorithm::Slink),
                "agnes" | "sahn" => out.push(SingleLinkAlgorithm::Agnes),
                "anderberg" => out.push(SingleLinkAlgorithm::Anderberg),
                "muellner" => out.push(SingleLinkAlgorithm::Muellner),
                "nnchain" | "nn-chain" => out.push(SingleLinkAlgorithm::NNChain),
                other => {
                    return Err(Box::<dyn Error>::from(format!("unknown algorithm '{other}'")));
                }
            }
        }
        Ok(out)
    } else {
        Ok(all.into_iter().filter(|a| !matches!(a, SingleLinkAlgorithm::Agnes)).collect())
    }
}

fn run_single_link_algorithm(
    algorithm: SingleLinkAlgorithm,
    tree_kind: TreeKind,
    seed: u64,
    sample_size: usize,
    data: &TableWithDistance<f64, Vec<f64>, CountingDistance<Euclidean>, f64>,
) -> MergeHistory<f64> {
    match algorithm {
        SingleLinkAlgorithm::Agnes => {
            let condensed = CondensedDistanceMatrix::new_from_data(data);
            agnes(&condensed, SingleLinkage)
        }
        SingleLinkAlgorithm::Anderberg => {
            let condensed = CondensedDistanceMatrix::new_from_data(data);
            anderberg(&condensed, SingleLinkage)
        }
        SingleLinkAlgorithm::Muellner => {
            let condensed = CondensedDistanceMatrix::new_from_data(data);
            muellner(&condensed, SingleLinkage)
        }
        SingleLinkAlgorithm::NNChain => {
            let condensed = CondensedDistanceMatrix::new_from_data(data);
            nn_chain(&condensed, SingleLinkage)
        }
        SingleLinkAlgorithm::Boruvka => {
            let mut rng = StdRng::seed_from_u64(seed);
            match tree_kind {
                TreeKind::Vp => {
                    let tree = VPTree::new(data, sample_size, &mut rng);
                    boruvka_searchers_single_link(&tree, data)
                }
                TreeKind::Kd => {
                    let tree = KdTree::new(data, MaxVarianceSplit);
                    boruvka_searchers_single_link(&tree, data)
                }
            }
        }
        SingleLinkAlgorithm::HeapOfSearchers => {
            let mut rng = StdRng::seed_from_u64(seed);
            match tree_kind {
                TreeKind::Vp => {
                    let tree = VPTree::new(data, sample_size, &mut rng);
                    heap_of_searchers_single_link(&tree, data)
                }
                TreeKind::Kd => {
                    let tree = KdTree::new(data, MaxVarianceSplit);
                    heap_of_searchers_single_link(&tree, data)
                }
            }
        }
        SingleLinkAlgorithm::RestartingSearch => {
            let mut rng = StdRng::seed_from_u64(seed);
            match tree_kind {
                TreeKind::Vp => {
                    let tree = VPTree::new(data, sample_size, &mut rng);
                    restarting_search_single_link(&tree, data)
                }
                TreeKind::Kd => {
                    let tree = KdTree::new(data, MaxVarianceSplit);
                    restarting_search_single_link(&tree, data)
                }
            }
        }
        SingleLinkAlgorithm::BufferedSearch { slack } => {
            let mut rng = StdRng::seed_from_u64(seed);
            match tree_kind {
                TreeKind::Vp => {
                    let tree = VPTree::new(data, sample_size, &mut rng);
                    buffered_search_single_link(&tree, data, slack)
                }
                TreeKind::Kd => {
                    let tree = KdTree::new(data, MaxVarianceSplit);
                    buffered_search_single_link(&tree, data, slack)
                }
            }
        }
        SingleLinkAlgorithm::LazyBufferedSearch { slack } => {
            let mut rng = StdRng::seed_from_u64(seed);
            match tree_kind {
                TreeKind::Vp => {
                    let tree = VPTree::new(data, sample_size, &mut rng);
                    lazy_buffered_search_single_link(&tree, data, slack)
                }
                TreeKind::Kd => {
                    let tree = KdTree::new(data, MaxVarianceSplit);
                    lazy_buffered_search_single_link(&tree, data, slack)
                }
            }
        }
        SingleLinkAlgorithm::Slink => slink(data),
    }
}

fn summarize_cluster_sizes(labels: &[usize]) -> (BTreeMap<usize, usize>, usize) {
    let cluster_sizes = labels.iter().fold(BTreeMap::new(), |mut m, &lbl| {
        *m.entry(lbl).or_insert(0) += 1;
        m
    });
    (cluster_sizes, 0)
}

// helper functions copied/ported from the regression support tests; these
// turn an extracted hierarchy into flat labels with roughly `min_clusters`
// groups.  using the simplified hierarchy allows us to handle ties and
// spurious cuts more gracefully than the naive cut-by-count.

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
    history: &MergeHistory<f64>, n: usize, min_clusters: usize,
) -> Vec<usize> {
    // Minpts 10, to give more meaningful clustering structure for comparison.
    let extracted = extract_simplified_hierarchy(history, None, 2);
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

#[derive(Clone, Copy)]
enum SingleLinkAlgorithm {
    Boruvka,
    HeapOfSearchers,
    RestartingSearch,
    BufferedSearch { slack: usize },
    LazyBufferedSearch { slack: usize },
    Slink,
    Agnes,
    Anderberg,
    Muellner,
    NNChain,
}

impl SingleLinkAlgorithm {
    fn label(self) -> String {
        match self {
            SingleLinkAlgorithm::Boruvka => "boruvka_searchers".to_string(),
            SingleLinkAlgorithm::HeapOfSearchers => "heap_of_searchers".to_string(),
            SingleLinkAlgorithm::RestartingSearch => "restarting_search".to_string(),
            SingleLinkAlgorithm::BufferedSearch { slack } => {
                format!("buffered_search(slack={slack})")
            }
            SingleLinkAlgorithm::LazyBufferedSearch { slack } => {
                format!("lazy_buffered_search(slack={slack})")
            }
            SingleLinkAlgorithm::Slink => "slink".to_string(),
            SingleLinkAlgorithm::Agnes => "agnes".to_string(),
            SingleLinkAlgorithm::Anderberg => "anderberg".to_string(),
            SingleLinkAlgorithm::Muellner => "muellner".to_string(),
            SingleLinkAlgorithm::NNChain => "nn_chain".to_string(),
        }
    }
}

fn parse_positive_usize(args: &mut std::env::Args, flag: &str) -> Result<usize, Box<dyn Error>> {
    let value = args.next().ok_or_else(|| missing_value_error(flag))?;
    let parsed = value.parse::<usize>().map_err(|_| positive_integer_error(flag))?;
    if parsed == 0 {
        return Err(Box::<dyn Error>::from(format!("{flag} must be greater than 0")));
    }
    Ok(parsed)
}

fn parse_seed(args: &mut std::env::Args, flag: &str) -> Result<u64, Box<dyn Error>> {
    let value = args.next().ok_or_else(|| missing_value_error(flag))?;
    let parsed = value.parse::<u64>().map_err(|_| non_negative_integer_error(flag))?;
    Ok(parsed)
}

fn usage_error() -> Box<dyn Error> { Box::<dyn Error>::from(USAGE) }

fn missing_value_error(flag: &str) -> Box<dyn Error> {
    Box::<dyn Error>::from(format!("missing value for {flag}"))
}

fn positive_integer_error(flag: &str) -> Box<dyn Error> {
    Box::<dyn Error>::from(format!("{flag} must be a positive integer"))
}

fn non_negative_integer_error(flag: &str) -> Box<dyn Error> {
    Box::<dyn Error>::from(format!("{flag} must be a non-negative integer"))
}
