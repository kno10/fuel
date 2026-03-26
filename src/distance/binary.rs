use num_traits::AsPrimitive;

use crate::Float;
use crate::distance::DistanceFunction;

pub(crate) fn bool_counts<N: Float>(a: &[N], b: &[N]) -> (u32, u32, u32, u32) {
    a.iter().zip(b.iter()).fold((0, 0, 0, 0), |(n00, n01, n10, n11), (x, y)| {
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

/// Hamming distance (binary vectors):
/// $$d_H(a,b)=\frac{1}{n}\sum_{i=1}^n [a_i \neq b_i]$$
pub fn hamming_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let n = n00 + n01 + n10 + n11;
    if n == 0 { F::zero() } else { (n01 + n10).as_() / n.as_() }
}

/// Matching distance (binary vectors, equivalent to Hamming):
/// $$d_M(a,b)=d_H(a,b)$$
pub fn matching_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    hamming_distance(a, b)
}

/// Jaccard distance:
/// $$d_J(a,b)=\frac{n_{01}+n_{10}}{n_{11}+n_{01}+n_{10}}$$
pub fn jaccard_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (_n00, n01, n10, n11) = bool_counts(a, b);
    let denominator = n11 + n01 + n10;
    if denominator == 0 { F::zero() } else { (n01 + n10).as_() / denominator.as_() }
}

/// Dice distance:
/// $$d_D(a,b)=\frac{n_{01}+n_{10}}{2n_{11}+n_{01}+n_{10}}$$
pub fn dice_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (_n00, n01, n10, n11) = bool_counts(a, b);
    let denominator = 2 * n11 + n01 + n10;
    if denominator == 0 { F::zero() } else { (n01 + n10).as_() / denominator.as_() }
}

/// Kulsinski distance:
/// $$d_K(a,b)=\frac{n_{01}+n_{10}-n_{11}+n}{n_{01}+n_{10}+n}$$
pub fn kulsinski_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let n = n00 + n01 + n10 + n11;
    let denominator = n01 + n10 + n;
    if denominator == 0 { F::zero() } else { (n01 + n10 - n11 + n).as_() / denominator.as_() }
}

/// Roger--Stanimoto distance:
/// $$d_{RS}(a,b)=\frac{2(n_{01}+n_{10})}{n_{11}+n_{00}+2(n_{01}+n_{10})}$$
pub fn roger_stanimoto_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let numerator = 2 * (n01 + n10);
    let denominator = n11 + n00 + numerator;
    if denominator == 0 { F::zero() } else { numerator.as_() / denominator.as_() }
}

/// Russell--Rao distance:
/// $$d_{RR}(a,b)=\frac{n-n_{11}}{n}$$
pub fn russell_rao_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (n00, n01, n10, n11) = bool_counts(a, b);
    let n = n00 + n01 + n10 + n11;
    if n == 0 { F::zero() } else { (n - n11).as_() / n.as_() }
}

/// Sokal--Michener distance (equivalent to Roger--Stanimoto):
/// $$d_{SM}(a,b)=d_{RS}(a,b)$$
pub fn sokal_michener_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    roger_stanimoto_distance(a, b)
}

/// Sokal--Sneath distance:
/// $$d_{SS}(a,b)=\frac{2(n_{01}+n_{10})}{n_{11}+2(n_{01}+n_{10})}$$
pub fn sokal_sneath_distance<N: Float, F: Float + 'static>(a: &[N], b: &[N]) -> F
where
    u32: AsPrimitive<F>,
{
    let (_n00, n01, n10, n11) = bool_counts(a, b);
    let numerator = 2 * (n01 + n10);
    let denominator = n11 + numerator;
    if denominator == 0 { F::zero() } else { numerator.as_() / denominator.as_() }
}

#[derive(Debug, Clone, Copy, Default)]
/// Hamming distance strategy (binary mismatches per dimension).
pub struct Hamming;

impl<N: Float> DistanceFunction<[N], f32> for Hamming
where
    u32: AsPrimitive<f32>,
{
    fn distance(&self, a: &[N], b: &[N]) -> f32 { hamming_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}

#[derive(Debug, Clone, Copy, Default)]
/// Matching distance strategy (binary matches vs mismatches).
pub struct Matching;

impl<N: Float> DistanceFunction<[N], f32> for Matching
where
    u32: AsPrimitive<f32>,
{
    fn distance(&self, a: &[N], b: &[N]) -> f32 { matching_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}

#[derive(Debug, Clone, Copy, Default)]
/// Jaccard similarity-derived distance strategy for binary vectors.
pub struct Jaccard;

impl<N: Float> DistanceFunction<[N], f32> for Jaccard
where
    u32: AsPrimitive<f32>,
{
    fn distance(&self, a: &[N], b: &[N]) -> f32 { jaccard_distance(a, b) }

    fn is_metric(&self) -> bool { true }
}

#[derive(Debug, Clone, Copy, Default)]
/// Dice distance strategy for binary set similarity.
pub struct Dice;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for Dice
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F { dice_distance(a, b) }
}

#[derive(Debug, Clone, Copy, Default)]
/// Kulsinski binary distance strategy emphasizing mismatches.
pub struct Kulsinski;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for Kulsinski
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F { kulsinski_distance(a, b) }
}

#[derive(Debug, Clone, Copy, Default)]
/// Roger-Stanimoto binary similarity-based distance strategy.
pub struct RogerStanimoto;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for RogerStanimoto
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F { roger_stanimoto_distance(a, b) }
}

#[derive(Debug, Clone, Copy, Default)]
/// Russell-Rao binary mismatch distance strategy.
pub struct RussellRao;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for RussellRao
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F { russell_rao_distance(a, b) }
}

#[derive(Debug, Clone, Copy, Default)]
/// Sokal-Michener binary distance strategy.
pub struct SokalMichener;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for SokalMichener
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F { sokal_michener_distance(a, b) }
}

#[derive(Debug, Clone, Copy, Default)]
/// Sokal-Sneath binary distance strategy.
pub struct SokalSneath;

impl<N: Float, F: Float + 'static> DistanceFunction<[N], F> for SokalSneath
where
    u32: AsPrimitive<F>,
{
    fn distance(&self, a: &[N], b: &[N]) -> F { sokal_sneath_distance(a, b) }
}
