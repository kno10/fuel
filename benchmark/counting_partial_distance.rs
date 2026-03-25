use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fuel::Float;
use fuel::distance::{DistanceFunction, PartialDistance};

#[derive(Debug, Clone)]
pub struct CountingPartialDistance<M> {
    counter: Arc<AtomicU64>,
    inner: M,
}

impl<M> CountingPartialDistance<M> {
    pub fn new(inner: M) -> Self { Self { counter: Arc::new(AtomicU64::new(0)), inner } }

    pub fn count(&self) -> u64 { self.counter.load(Ordering::Relaxed) }
}

impl<M, F> PartialDistance<F, F> for CountingPartialDistance<M>
where
    F: Float,
    M: PartialDistance<F, F>,
{
    fn axis_distance(&self, delta: F) -> F { self.inner.axis_distance(delta) }

    fn combine_axis_distances(&self, a: F, b: F) -> F { self.inner.combine_axis_distances(a, b) }
}

impl<M, F> DistanceFunction<[F], F> for CountingPartialDistance<M>
where
    F: Float,
    M: DistanceFunction<[F], F>,
{
    fn distance(&self, a: &[F], b: &[F]) -> F {
        self.counter.fetch_add(1, Ordering::Relaxed);
        self.inner.distance(a, b)
    }
}

impl<M, F> DistanceFunction<Vec<F>, F> for CountingPartialDistance<M>
where
    F: Float,
    M: DistanceFunction<Vec<F>, F>,
{
    fn distance(&self, a: &Vec<F>, b: &Vec<F>) -> F {
        self.counter.fetch_add(1, Ordering::Relaxed);
        self.inner.distance(a, b)
    }
}
