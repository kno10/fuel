use crate::Float;
use crate::distance::DistanceFunction;

/// # Panics
///
/// Panics if either input slice does not contain exactly two values
/// (`[latitude, longitude]`).
/// Haversine distance (spherical):
/// $$d_{Hav}(a,b)=2r\arcsin\left(\sqrt{\sin^2(\frac{\Delta\phi}{2})+\cos\phi_1\cos\phi_2\sin^2(\frac{\Delta\lambda}{2})}\right)$$
pub fn haversine_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    assert!(a.len() == 2 && b.len() == 2, "Haversine distance expects [latitude, longitude] pairs");

    let lat1: F = a[0].to_float::<F>();
    let lon1: F = a[1].to_float::<F>();
    let lat2: F = b[0].to_float::<F>();
    let lon2: F = b[1].to_float::<F>();

    let dlat = (lat2 - lat1) * F::half();
    let dlon = (lon2 - lon1) * F::half();
    let sin_dlat = dlat.sin();
    let sin_dlon = dlon.sin();

    let h = sin_dlat * sin_dlat + lat1.cos() * lat2.cos() * sin_dlon * sin_dlon;
    F::two() * h.sqrt().asin()
}

#[derive(Debug, Clone, Copy, Default)]
/// Haversine distance strategy for spherical coordinates.
pub struct Haversine;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for Haversine {
    fn distance(&self, a: &[N], b: &[N]) -> F { haversine_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}
