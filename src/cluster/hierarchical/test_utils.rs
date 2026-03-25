/// Utilities used exclusively by hierarchical clustering unit tests.
#[cfg(test)]
use crate::distance::DistanceFunction;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScalarDistance;

impl DistanceFunction<[f64], f64> for ScalarDistance {
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let ai = a.first().copied().unwrap_or(0.0);
        let bi = b.first().copied().unwrap_or(0.0);
        (ai - bi).abs()
    }
}
