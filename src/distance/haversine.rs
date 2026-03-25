use crate::Float;
use crate::distance::{DistanceFunction, DistanceMetric};

/// # Panics
///
/// Panics if either input slice does not contain exactly two values
/// (`[latitude, longitude]`).
pub fn haversine_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    assert!(a.len() == 2 && b.len() == 2, "Haversine distance expects [latitude, longitude] pairs");

    let lat1: F = a[0].to_float::<F>();
    let lon1: F = a[1].to_float::<F>();
    let lat2: F = b[0].to_float::<F>();
    let lon2: F = b[1].to_float::<F>();

    let half = F::one() / (F::one() + F::one());
    let dlat = (lat2 - lat1) * half;
    let dlon = (lon2 - lon1) * half;
    let sin_dlat = dlat.sin();
    let sin_dlon = dlon.sin();

    let h = sin_dlat * sin_dlat + lat1.cos() * lat2.cos() * sin_dlon * sin_dlon;
    (F::one() + F::one()) * h.sqrt().asin()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HaversineDistance;

impl<N: Float, F: Float + 'static> DistanceMetric<[N], F> for HaversineDistance {}

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for HaversineDistance {
    fn distance(&self, a: &[N], b: &[N]) -> F { haversine_distance(a, b) }
}
