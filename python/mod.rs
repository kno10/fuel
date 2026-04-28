use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::SeedableRng;
use rand_pcg::Pcg32;

use crate::Float;
use crate::distance::{
    Arccosine, BrayCurtis, Canberra, Chebyshev, Chi, ChiSquared, Clark, Cosine, DistanceFunction,
    Euclidean, Hellinger, HistogramIntersection, Jeffrey, JensenShannon, Manhattan,
    SquaredEuclidean,
};

// FIXME: inline the make_rng function in all the call sites instead! Do not just remove this comment.
fn make_rng(seed: Option<u64>) -> Pcg32 { Pcg32::seed_from_u64(seed.unwrap_or(0)) }

/// Parse a distance name into a boxed dynamic distance function.
///
/// Accepted names (case-insensitive):
/// euclidean / l2, sqeuclidean / squared_euclidean, manhattan / l1 / cityblock,
/// chebyshev / linf / chessboard, cosine, arccosine / angular,
/// canberra, braycurtis / bray_curtis, hellinger, clark, chi,
/// chi_squared / chisquared / chi2, jensen_shannon / jensenshannon / js,
/// jeffrey / jeffreys, histogram_intersection / intersection.
pub(super) fn parse_distance_fn<N>(dist: &str) -> PyResult<Box<dyn DistanceFunction<[N], N> + Sync>>
where
    N: Float,
{
    match dist.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Box::new(Euclidean)),
        "sqeuclidean" | "squared_euclidean" => Ok(Box::new(SquaredEuclidean)),
        "manhattan" | "l1" | "cityblock" => Ok(Box::new(Manhattan)),
        "chebyshev" | "linf" | "chessboard" => Ok(Box::new(Chebyshev)),
        "cosine" => Ok(Box::new(Cosine)),
        "arccosine" | "angular" => Ok(Box::new(Arccosine)),
        "canberra" => Ok(Box::new(Canberra)),
        "braycurtis" | "bray_curtis" => Ok(Box::new(BrayCurtis)),
        "hellinger" => Ok(Box::new(Hellinger)),
        "clark" => Ok(Box::new(Clark)),
        "chi" => Ok(Box::new(Chi)),
        "chi_squared" | "chisquared" | "chi2" => Ok(Box::new(ChiSquared)),
        "jensen_shannon" | "jensenshannon" | "js" => Ok(Box::new(JensenShannon)),
        "jeffrey" | "jeffreys" => Ok(Box::new(Jeffrey)),
        "histogram_intersection" | "intersection" => Ok(Box::new(HistogramIntersection)),
        other => Err(PyValueError::new_err(format!(
            "unknown distance '{}', valid options are: euclidean, sqeuclidean, manhattan, \
             chebyshev, cosine, arccosine, canberra, braycurtis, hellinger, clark, chi, \
             chi_squared, jensen_shannon, jeffrey, histogram_intersection",
            other
        ))),
    }
}

mod em;
mod hierarchical;
mod kmeans;
mod outlier;
mod sparse;
mod spherical;

#[pymodule]
#[pyo3(module = "fuel", name = "_fuel")]
fn _fuel<'py>(_py: Python<'py>, m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    kmeans::register(m)?;
    spherical::register(m)?;
    em::register(m)?;
    hierarchical::register(m)?;
    outlier::register(m)?;
    Ok(())
}
