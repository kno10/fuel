use num_traits::{AsPrimitive, Float, ToPrimitive};

use super::{DistanceFunction, DistanceMetric};

pub fn bool_counts<N: Float + ToPrimitive>(a: &[N], b: &[N]) -> (u32, u32, u32, u32) {
    a.iter()
        .zip(b.iter())
        .fold((0, 0, 0, 0), |(n00, n01, n10, n11), (x, y)| {
            let xb = *x != N::zero();
            let yb = *y != N::zero();
            match (xb, yb) {
                (false, false) => (n00 + 1, n01, n10, n11),
                (false, true) => (n00, n01 + 1, n10, n11),
                (true, false) => (n00, n01, n10 + 1, n11),
                (true, true) => (n00, n01, n10, n11 + 1),
            }
        })
}

fn cast_unchecked<F: Float + 'static>(value: u32) -> F
where
    u32: AsPrimitive<F>,
{
    value.as_()
}

pub fn hamming_distance<N: Float + ToPrimitive, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let n = n00 + n01 + n10 + n11;
    if n == 0 {
        F::zero()
    } else {
        cast_unchecked::<F>(n01 + n10) / cast_unchecked::<F>(n)
    }
}

pub fn matching_distance<N: Float + ToPrimitive, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    hamming_distance(a, b)
}

pub fn jaccard_distance<N: Float + ToPrimitive, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (_n00, n01, n10, n11) = bool_counts(a, b);
    let denominator = n11 + n01 + n10;
    if denominator == 0 {
        F::zero()
    } else {
        cast_unchecked::<F>(n01 + n10) / cast_unchecked::<F>(denominator)
    }
}

pub fn dice_distance<N: Float + ToPrimitive, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (_n00, n01, n10, n11) = bool_counts(a, b);
    let denominator = 2 * n11 + n01 + n10;
    if denominator == 0 {
        F::zero()
    } else {
        cast_unchecked::<F>(n01 + n10) / cast_unchecked::<F>(denominator)
    }
}

pub fn kulsinski_distance<N: Float + ToPrimitive, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let n = n00 + n01 + n10 + n11;
    let denominator = n01 + n10 + n;
    if denominator == 0 {
        F::zero()
    } else {
        cast_unchecked::<F>(n01 + n10 - n11 + n) / cast_unchecked::<F>(denominator)
    }
}

pub fn roger_stanimoto_distance<N: Float + ToPrimitive, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let numerator = 2 * (n01 + n10);
    let denominator = n11 + n00 + numerator;
    if denominator == 0 {
        F::zero()
    } else {
        cast_unchecked::<F>(numerator) / cast_unchecked::<F>(denominator)
    }
}

pub fn russell_rao_distance<N: Float + ToPrimitive, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let n = n00 + n01 + n10 + n11;
    if n == 0 {
        F::zero()
    } else {
        cast_unchecked::<F>(n - n11) / cast_unchecked::<F>(n)
    }
}

pub fn sokal_michener_distance<N: Float + ToPrimitive, F: Float + 'static>(
    a: &[N],
    b: &[N],
) -> F
where
    u32: AsPrimitive<F>,
{
    roger_stanimoto_distance(a, b)
}

pub fn sokal_sneath_distance<N: Float + ToPrimitive, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (_n00, n01, n10, n11) = bool_counts(a, b);
    let numerator = 2 * (n01 + n10);
    let denominator = n11 + numerator;
    if denominator == 0 {
        F::zero()
    } else {
        cast_unchecked::<F>(numerator) / cast_unchecked::<F>(denominator)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HammingDistance;

impl<N: Float + ToPrimitive> DistanceMetric<[N]> for HammingDistance {}

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F> for HammingDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        hamming_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MatchingDistance;

impl<N: Float + ToPrimitive> DistanceMetric<[N]> for MatchingDistance {}

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F>
    for MatchingDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        matching_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct JaccardDistance;

impl<N: Float + ToPrimitive> DistanceMetric<[N]> for JaccardDistance {}

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F> for JaccardDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        jaccard_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DiceDistance;

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F> for DiceDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        dice_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KulsinskiDistance;

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F> for KulsinskiDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        kulsinski_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RogerStanimotoDistance;

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F>
    for RogerStanimotoDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        roger_stanimoto_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RussellRaoDistance;

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F> for RussellRaoDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        russell_rao_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SokalMichenerDistance;

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F>
    for SokalMichenerDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        sokal_michener_distance(a, b)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SokalSneathDistance;

impl<N: Float + ToPrimitive, F: Float + 'static> DistanceFunction<[N], F> for SokalSneathDistance
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F {
        sokal_sneath_distance(a, b)
    }
}
