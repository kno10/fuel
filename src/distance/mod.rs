//! Distance metrics and divergences for numeric vectors.
//!
//! Rich metric list with formula, struct mapping, and characteristic notes:
//!
//! ### Binary (boolean set) distances
//! | Name | Function | Struct | Formula | Metric | Notes |
//! |---|---|---|---|---|---|
//! | Hamming | `hamming_distance` | `Hamming` | `(n01 + n10)/n` | yes | mismatch fraction |
//! | Matching | `matching_distance` | `Matching` | `(n01 + n10)/n` | yes | same as Hamming |
//! | Jaccard | `jaccard_distance` | `Jaccard` | `(n01+n10)/(n11+n01+n10)` | yes | set similarity complement |
//! | Dice | `dice_distance` | `Dice` | `(n01+n10)/(2n11+n01+n10)` | no | F1-based form |
//! | Kulsinski | `kulsinski_distance` | `Kulsinski` | `(n01+n10-n11+n)/(n01+n10+n)` | no | asymmetric bias |
//! | Roger-Stanimoto | `roger_stanimoto_distance` | `RogerStanimoto` | `2(n01+n10)/(n11+n00+2(n01+n10))` | yes | metric |
//! | Russell-Rao | `russell_rao_distance` | `RussellRao` | `(n-n11)/n` | yes | binary dissimilarity |
//! | Sokal-Michener | `sokal_michener_distance` | `SokalMichener` | `2(n01+n10)/(n11+n00+2(n01+n10))` | yes | same as Roger-Stanimoto |
//! | Sokal-Sneath | `sokal_sneath_distance` | `SokalSneath` | `2(n01+n10)/(n11+2(n01+n10))` | no | relaxed complement |
//!
//! ### Set & overlap distances
//! | Name | Function | Struct | Formula | Metric | Notes |
//! |---|---|---|---|---|---|
//! | Bray-Curtis | `braycurtis_distance` | `BrayCurtis` | `sum|a-b| / sum|a+b|` | no | normalized L1 |
//! | Canberra | `canberra_distance` | `Canberra` | `sum |a-b|/(|a|+|b|)` | yes | per-dim scaling |
//! | Clark | `clark_distance` | `Clark` | `sqrt(sum((a-b)/(a+b))^2)` | yes | normalized L2 |
//! | Histogram Intersection | `histogram_intersection_distance` | `HistogramIntersection` | `1 - sum(min(a,b))/sum(a)` | no | overlap similarity |
//!
//! ### Lᵖ metrics
//! | Name | Function | Struct | Formula | Metric | Notes |
//! |---|---|---|---|---|---|
//! | Manhattan | `manhattan_distance` | `Manhattan` | `sum |a-b|` | yes | L1 norm |
//! | Euclidean | `euclidean_distance` | `Euclidean` | `sqrt(sum (a-b)^2)` | yes | L2 norm |
//! | Squared Euclidean | `squared_euclidean_distance` | `SquaredEuclidean` | `sum (a-b)^2` | yes | faster for comparison |
//! | Minkowski | `minkowski_distance` | `Minkowski` | `(sum |a-b|^p)^(1/p)` | yes if p>=1 | general Lp |
//! | Chebyshev | `chebyshev_distance` | `Chebyshev` | `max |a-b|` | yes | L∞ norm |
//!
//! ### Angular distances
//! | Name | Function | Struct | Formula | Metric | Notes |
//! |---|---|---|---|---|---|
//! | Cosine | `cosine_distance` | `Cosine` | `1 - (a·b)/(||a|| ||b||)` | no | similarity complement |
//! | Arccosine | `arccosine_distance` | `Arccosine` | `arccos((a·b)/(||a|| ||b||))` | yes | angle metric |
//! | Haversine | `haversine_distance` | `Haversine` | sphere great-circle | yes | geographic |
//!
//! ### Probabilistic distances
//! | Name | Function | Struct | Formula | Metric | Notes |
//! |---|---|---|---|---|---|
//! | Hellinger | `hellinger_distance` | `Hellinger` | `sqrt(1 - sum sqrt(a b))` | yes | distribution metric |
//! | Chi | `chi_distance` | `Chi` | `sqrt(sum (a-b)^2/(a+b))` | yes | divergence scale |
//! | Chi Squared | `chi_squared_distance` | `ChiSquared` | `1/2 sum (a-b)^2/(a+b)` | yes | normalized Chi-sq |

//! This module exports both the function and struct API in direct correspondence.
//!
mod binary;
mod braycurtis;
mod canberra;
mod chebyshev;
mod chi;
mod chi_squared;
mod clark;
mod cosine;
mod euclidean;
mod haversine;
mod hellinger;
mod histogram_intersection;
mod inner_product;
mod jeffrey;
mod jensen_shannon;
mod manhattan;
mod minkowski;
mod partial;
mod squared_euclidean;

pub trait DistanceFunction<T: ?Sized, F: Float> {
    fn distance(&self, a: &T, b: &T) -> F;

    fn is_metric(&self) -> bool { false }
}

impl<T: ?Sized, D, F: Float> DistanceFunction<T, F> for Box<D>
where
    D: DistanceFunction<T, F>,
{
    fn distance(&self, a: &T, b: &T) -> F { (**self).distance(a, b) }

    fn is_metric(&self) -> bool { (**self).is_metric() }
}

pub use binary::{
    Dice, Hamming, Jaccard, Kulsinski, Matching, RogerStanimoto, RussellRao, SokalMichener,
    SokalSneath, dice_distance, hamming_distance, jaccard_distance, kulsinski_distance,
    matching_distance, roger_stanimoto_distance, russell_rao_distance, sokal_michener_distance,
    sokal_sneath_distance,
};
pub use braycurtis::{BrayCurtis, braycurtis_distance};
pub use canberra::{Canberra, canberra_distance};
pub use chebyshev::{Chebyshev, chebyshev_distance};
pub use chi::{Chi, chi_distance};
pub use chi_squared::{ChiSquared, chi_squared_distance};
pub use clark::{Clark, clark_distance};
pub use cosine::{Arccosine, Cosine, arccosine_distance, cosine_distance};
pub use euclidean::{Euclidean, euclidean_distance};
pub use haversine::{Haversine, haversine_distance};
pub use hellinger::{Hellinger, hellinger_distance};
pub use histogram_intersection::{HistogramIntersection, histogram_intersection_distance};
pub use inner_product::dot;
pub use jeffrey::{Jeffrey, jeffrey_divergence};
pub use jensen_shannon::{JensenShannon, jensen_shannon_divergence};
pub use manhattan::{Manhattan, manhattan_distance};
pub use minkowski::{Minkowski, minkowski_distance};
pub use partial::PartialDistance;
pub use squared_euclidean::{SquaredEuclidean, squared_euclidean_distance};

use crate::Float;
