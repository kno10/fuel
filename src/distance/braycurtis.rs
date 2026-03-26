use crate::Float;
use crate::distance::DistanceFunction;

/// Bray--Curtis distance:
/// $$d_{BC}(a,b)=\frac{\sum_i |a_i-b_i|}{\sum_i |a_i+b_i|}$$
pub fn braycurtis_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F {
    let numerator = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let left: F = (*x).to_float::<F>();
            let right: F = (*y).to_float::<F>();
            (left - right).abs()
        })
        .fold(F::zero(), |acc, value| acc + value);
    let denominator = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let left: F = (*x).to_float::<F>();
            let right: F = (*y).to_float::<F>();
            (left + right).abs()
        })
        .fold(F::zero(), |acc, value| acc + value);

    if denominator == F::zero() { F::zero() } else { numerator / denominator }
}

#[derive(Debug, Clone, Copy, Default)]
/// Bray-Curtis distance strategy (normalized L1).
pub struct BrayCurtis;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for BrayCurtis {
    fn distance(&self, a: &[N], b: &[N]) -> F { braycurtis_distance(a, b) }
}
