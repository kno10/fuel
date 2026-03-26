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
}

impl<T: ?Sized, D, F: Float> DistanceFunction<T, F> for Box<D>
where
    D: DistanceFunction<T, F>,
{
    fn distance(&self, a: &T, b: &T) -> F { (**self).distance(a, b) }
}
// Marker trait for distance metrics that satisfy the triangle inequality.
pub trait DistanceMetric<T: ?Sized, F: Float>: DistanceFunction<T, F> {}

pub use binary::{
    DiceDistance, HammingDistance, JaccardDistance, KulsinskiDistance, MatchingDistance,
    RogerStanimotoDistance, RussellRaoDistance, SokalMichenerDistance, SokalSneathDistance,
};
pub use braycurtis::BrayCurtisDistance;
pub use canberra::CanberraDistance;
pub use chebyshev::ChebyshevDistance;
pub use chi::{ChiDistance, chi_distance};
pub use chi_squared::{ChiSquaredDistance, chi_squared_distance};
pub use clark::ClarkDistance;
pub use cosine::{ArccosineDistance, CosineDistance, arccosine_distance, cosine_distance};
pub use euclidean::EuclideanDistance;
pub use haversine::HaversineDistance;
pub use hellinger::{HellingerDistance, hellinger_distance};
pub use histogram_intersection::{HistogramIntersectionDistance, histogram_intersection_distance};
pub use inner_product::dot;
pub use jeffrey::{JeffreyDistance, jeffrey_divergence};
pub use jensen_shannon::{JensenShannonDistance, jensen_shannon_divergence};
pub use manhattan::ManhattanDistance;
pub use minkowski::MinkowskiDistance;
pub use partial::PartialDistance;
pub use squared_euclidean::{SquaredEuclideanDistance, squared_euclidean_distance};

use crate::Float;
