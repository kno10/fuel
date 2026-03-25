use crate::Float;
use crate::distance::DistanceFunction;

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
pub struct BrayCurtisDistance;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for BrayCurtisDistance {
    fn distance(&self, a: &[N], b: &[N]) -> F { braycurtis_distance(a, b) }
}
