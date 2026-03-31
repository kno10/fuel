mod common;

use std::error::Error;
use std::time::Instant;

use common::{CountingDistance, read_numeric_data};
use fuel::TableWithDistance;
use fuel::distance::Euclidean;
use fuel::outlier::{distance_from_center, distance_from_origin, random, zero};
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

    let csv_path = args.next().ok_or(
        "usage: cargo run --features benchmark --bin baseline -- <csv_path> <mode> [seed]",
    )?;

    let mode = args.next().ok_or("missing mode (origin|center|random|zero)")?;

    let seed: u64 =
        args.next().map(|s| s.parse().expect("seed must be an integer")).unwrap_or(RNG_SEED);

    let rows = read_numeric_data(&csv_path)?;
    if rows.is_empty() {
        return Err("CSV must contain at least one row".into());
    }

    // we build the tree only to have a uniform interface; the baseline
    // algorithms do not depend on it, but it exercises indexing overhead
    let distance = CountingDistance::new(Euclidean);
    let data: TableWithDistance<f64, Vec<f64>, CountingDistance<Euclidean>, f64> =
        TableWithDistance::with_distance(&rows, distance.clone());
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let sample_size = rows.len();

    let start = Instant::now();
    let _tree = fuel::search::vptree::VPTree::new(&data, sample_size, &mut rng);
    let distance_count_after_index = distance.count();

    let scores = match mode.as_str() {
        "origin" => distance_from_origin(&data),
        "center" => distance_from_center(&data),
        "random" => random(&data, seed),
        "zero" => zero(&data),
        other => return Err(format!("unknown mode: {}", other).into()),
    };

    let dist_count = distance.count();
    let elapsed = start.elapsed();

    let avg_score = scores.scores.iter().copied().sum::<f64>() / scores.scores.len() as f64;

    println!("time_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("avg_score={avg_score:.12}");
    println!("distance_count_after_index={distance_count_after_index}");
    println!("dist_count={dist_count}");

    Ok(())
}
