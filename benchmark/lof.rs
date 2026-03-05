mod counting_distance;
mod data_loading;

use std::error::Error;
use std::sync::atomic::Ordering;
use std::time::Instant;

use counting_distance::CountingEuclideanDistance;
use data_loading::read_numeric_data;
use hacs::{MatrixDataAccess, VPTree, lof_outlier_scores};
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
        .ok_or("usage: cargo run --features benchmark --bin lof_benchmark -- <csv_path> <k>")?;

    let k: usize = args
        .next()
        .ok_or("missing k")?
        .parse()
        .map_err(|_| "k must be a positive integer")?;

    if k == 0 {
        return Err("k must be greater than 0".into());
    }

    let rows = read_numeric_data(&csv_path)?;
    if rows.len() < 2 {
        return Err("CSV must contain at least two rows".into());
    }

    let distance = CountingEuclideanDistance::new();
    let distance_count = distance.counter();
    let data = MatrixDataAccess::with_distance(&rows, distance);
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let sample_size = rows.len();

    let start = Instant::now();
    let tree = VPTree::new(&data, sample_size, &mut rng);
    let distance_count_after_index = distance_count.load(Ordering::Relaxed);
    let scores = lof_outlier_scores(&tree, &data, k);
    let distance_count_after_algorithm = distance_count.load(Ordering::Relaxed);
    let elapsed = start.elapsed();

    let avg_score = scores.iter().map(|entry| entry.score).sum::<f64>() / scores.len() as f64;

    println!("time_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("avg_score={avg_score:.12}");
    println!("distance_count_after_index={distance_count_after_index}");
    println!("distance_count_after_algorithm={distance_count_after_algorithm}");

    Ok(())
}
