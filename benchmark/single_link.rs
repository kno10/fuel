mod data_loading;

use std::error::Error;
use std::time::Instant;

use hacs::{
    DataAccess, EuclideanDistance, MatrixDataAccess, MergeHistory, VPTree,
    boruvka_searchers_single_link, buffered_search_single_link, heap_of_searchers_single_link,
    restarting_search_single_link, slink,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

use data_loading::read_numeric_data_with_limit;

const DEFAULT_BUFFERED_SLACK: usize = 2;
const DEFAULT_TREE_SAMPLE: usize = 16;
const DEFAULT_VPTREE_SEED: u64 = 0xDEADBEEF;
const USAGE: &str = "usage: cargo run --features benchmark --bin single_link -- <csv_path> <n> [--tree-sample SIZE] [--buffered-slack SIZE] [--seed SEED]";

fn main() -> Result<(), Box<dyn Error>> {
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

    let mut tree_sample_size = DEFAULT_TREE_SAMPLE;
    let mut buffered_slack = DEFAULT_BUFFERED_SLACK;
    let mut seed = DEFAULT_VPTREE_SEED;

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--tree-sample" => {
                tree_sample_size = parse_positive_usize(&mut args, &flag)?;
            }
            "--buffered-slack" => {
                buffered_slack = parse_positive_usize(&mut args, &flag)?;
            }
            "--seed" => {
                seed = parse_seed(&mut args, &flag)?;
            }
            _ => {
                return Err(Box::<dyn Error>::from(format!(
                    "unexpected argument '{flag}'"
                )));
            }
        }
    }

    let mut rows = read_numeric_data_with_limit(&csv_path, Some(requested_rows))?;
    if rows.len() < requested_rows {
        return Err(format!(
            "CSV {csv_path} only contains {} rows but {} were requested",
            rows.len(),
            requested_rows
        )
        .into());
    }
    rows.truncate(requested_rows);
    let used_rows = rows.len();

    let dimension = rows.first().map(|row| row.len()).unwrap_or(0);
    let data: MatrixDataAccess<Vec<f64>, EuclideanDistance> =
        MatrixDataAccess::with_distance(&rows, EuclideanDistance);
    let sample_size = tree_sample_size.min(used_rows).max(1);

    let mut rng = StdRng::seed_from_u64(seed);
    let tree_start = Instant::now();
    let tree = VPTree::new(&data, sample_size, &mut rng);
    let tree_build_time = tree_start.elapsed();

    let condensed_start = Instant::now();
    let condensed = build_condensed_distances(&data);
    let condensed_time = condensed_start.elapsed();

    println!("dataset={csv_path}");
    println!("data_rows={used_rows}");
    println!("dimensions={dimension}");
    println!("tree_sample_size={sample_size}");
    println!("buffered_slack={buffered_slack}");
    println!("seed={seed}");
    println!(
        "tree_build_ms={:.3}",
        tree_build_time.as_secs_f64() * 1_000.0
    );
    println!(
        "condensed_matrix_ms={:.3}",
        condensed_time.as_secs_f64() * 1_000.0
    );

    let algorithms = [
        SingleLinkAlgorithm::Boruvka,
        SingleLinkAlgorithm::HeapOfSearchers,
        SingleLinkAlgorithm::RestartingSearch,
        SingleLinkAlgorithm::BufferedSearch {
            slack: buffered_slack,
        },
        SingleLinkAlgorithm::Slink,
    ];

    for algorithm in algorithms {
        let label = algorithm.label();
        let start = Instant::now();
        let history = algorithm.run(&tree, &data, &condensed, used_rows);
        let elapsed = start.elapsed();
        drop(history);
        println!(
            "algorithm={label}, time_ms={:.3}",
            elapsed.as_secs_f64() * 1_000.0
        );
    }

    Ok(())
}

fn build_condensed_distances(data: &MatrixDataAccess<'_, Vec<f64>, EuclideanDistance>) -> Vec<f64> {
    let n = data.size();
    let capacity = n.saturating_sub(1) * n / 2;
    let mut condensed = Vec::with_capacity(capacity);
    for i in 1..n {
        for j in 0..i {
            condensed.push(data.distance(i, j));
        }
    }
    condensed
}

#[derive(Clone, Copy)]
enum SingleLinkAlgorithm {
    Boruvka,
    HeapOfSearchers,
    RestartingSearch,
    BufferedSearch { slack: usize },
    Slink,
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
            SingleLinkAlgorithm::Slink => "slink".to_string(),
        }
    }

    fn run(
        self,
        tree: &VPTree,
        data: &MatrixDataAccess<'_, Vec<f64>, EuclideanDistance>,
        condensed: &[f64],
        n: usize,
    ) -> MergeHistory<f64> {
        match self {
            SingleLinkAlgorithm::Boruvka => boruvka_searchers_single_link(tree, data),
            SingleLinkAlgorithm::HeapOfSearchers => heap_of_searchers_single_link(tree, data),
            SingleLinkAlgorithm::RestartingSearch => restarting_search_single_link(tree, data),
            SingleLinkAlgorithm::BufferedSearch { slack } => {
                buffered_search_single_link(tree, data, slack)
            }
            SingleLinkAlgorithm::Slink => slink::<f64>(condensed, n),
        }
    }
}

fn parse_positive_usize(args: &mut std::env::Args, flag: &str) -> Result<usize, Box<dyn Error>> {
    let value = args.next().ok_or_else(|| missing_value_error(flag))?;
    let parsed = value
        .parse::<usize>()
        .map_err(|_| positive_integer_error(flag))?;
    if parsed == 0 {
        return Err(Box::<dyn Error>::from(format!(
            "{flag} must be greater than 0"
        )));
    }
    Ok(parsed)
}

fn parse_seed(args: &mut std::env::Args, flag: &str) -> Result<u64, Box<dyn Error>> {
    let value = args.next().ok_or_else(|| missing_value_error(flag))?;
    let parsed = value
        .parse::<u64>()
        .map_err(|_| non_negative_integer_error(flag))?;
    Ok(parsed)
}

fn usage_error() -> Box<dyn Error> {
    Box::<dyn Error>::from(USAGE)
}

fn missing_value_error(flag: &str) -> Box<dyn Error> {
    Box::<dyn Error>::from(format!("missing value for {flag}"))
}

fn positive_integer_error(flag: &str) -> Box<dyn Error> {
    Box::<dyn Error>::from(format!("{flag} must be a positive integer"))
}

fn non_negative_integer_error(flag: &str) -> Box<dyn Error> {
    Box::<dyn Error>::from(format!("{flag} must be a non-negative integer"))
}
