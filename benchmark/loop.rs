mod common;

use std::error::Error;
use std::time::Instant;

use common::{CountingDistance, read_numeric_data};
use fuel::TableWithDistance;
use fuel::distance::Euclidean;
use fuel::outlier::local_outlier_probabilities;
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

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);

    let csv_path = args
        .next()
        .ok_or("usage: cargo run --features benchmark --bin loop -- <csv_path> <k> [n_lambda]")?;

    let k: usize =
        args.next().ok_or("missing k")?.parse().map_err(|_| "k must be a positive integer")?;

    if k == 0 {
        return Err("k must be greater than 0".into());
    }

    let n_lambda: f64 = match args.next() {
        Some(value) => {
            value.parse().map_err(|_| "n_lambda must be a valid floating-point number")?
        }
        None => 2.0,
    };

    let rows = read_numeric_data(&csv_path)?;
    if rows.len() < 2 {
        return Err("CSV must contain at least two rows".into());
    }

    let distance = CountingDistance::new(Euclidean);
    let data: TableWithDistance<f64, Vec<f64>, CountingDistance<Euclidean>, f64> =
        TableWithDistance::with_distance(&rows, distance.clone());
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let sample_size = rows.len();

    let start = Instant::now();
    let tree = VPTree::new(&data, sample_size, &mut rng);
    let distance_count_after_index = distance.count();
    let scores = local_outlier_probabilities(&tree, &data, k, n_lambda);
    let dist_count = distance.count();
    let elapsed = start.elapsed();

    let avg_score = scores.scores.iter().copied().sum::<f64>() / scores.scores.len() as f64;

    println!("time_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("avg_score={avg_score:.12}");
    println!("distance_count_after_index={distance_count_after_index}");
    println!("dist_count={dist_count}");

    Ok(())
}
