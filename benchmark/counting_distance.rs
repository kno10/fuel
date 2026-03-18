use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fuel::distance::{DistanceFunction, EuclideanDistance};

#[derive(Debug, Clone, Default)]
pub struct CountingEuclideanDistance {
    counter: Arc<AtomicU64>,
}

impl CountingEuclideanDistance {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.counter)
    }
}

impl DistanceFunction<Vec<f64>, f64> for CountingEuclideanDistance {
    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        self.counter.fetch_add(1, Ordering::Relaxed);
        EuclideanDistance.distance(a, b)
    }
}
