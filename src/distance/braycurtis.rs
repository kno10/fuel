use num_traits::{AsPrimitive, Float, ToPrimitive};

use super::DistanceFunction;

pub fn braycurtis_distance<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F {
    let numerator = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let left: F = (*x).as_();
            let right: F = (*y).as_();
            (left - right).abs()
        })
        .fold(F::zero(), |acc, value| acc + value);
    let denominator = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let left: F = (*x).as_();
            let right: F = (*y).as_();
            (left + right).abs()
        })
        .fold(F::zero(), |acc, value| acc + value);

    if denominator == F::zero() {
        F::zero()
    } else {
        numerator / denominator
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BrayCurtisDistance;

impl<N: Float + ToPrimitive + AsPrimitive<F>, F: Float + 'static> DistanceFunction<[N], F>
    for BrayCurtisDistance
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        braycurtis_distance(a, b)
    }
}
