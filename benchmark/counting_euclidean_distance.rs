use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fuel::distance::DistanceFunction;

#[derive(Debug, Clone)]
pub struct CountingEuclideanDistance {
    counter: Arc<AtomicU64>,
}

impl CountingEuclideanDistance {
    pub fn new() -> Self { Self { counter: Arc::new(AtomicU64::new(0)) } }

    pub fn counter(&self) -> Arc<AtomicU64> { Arc::clone(&self.counter) }
}

impl DistanceFunction<Vec<f64>, f64> for CountingEuclideanDistance {
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        self.counter.fetch_add(1, Ordering::Relaxed);
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }
}

impl DistanceFunction<[f64], f64> for CountingEuclideanDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        self.counter.fetch_add(1, Ordering::Relaxed);
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }
}
