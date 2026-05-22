use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::SeedableRng;
use rand_pcg::Pcg32;

use crate::Float;
use crate::distance::{
    Arccosine, BrayCurtis, Canberra, Chebyshev, Chi, ChiSquared, Clark, Cosine, DistanceFunction,
    Euclidean, Hellinger, HistogramIntersection, Jeffrey, JensenShannon, Manhattan,
    PartialDistance, SquaredEuclidean,
};

pub(super) trait KdDistanceFunction<N: Float, F: Float>:
    DistanceFunction<[N], F> + PartialDistance<N, F>
{
}
impl<T, N, F> KdDistanceFunction<N, F> for T
where
    N: Float,
    F: Float,
    T: DistanceFunction<[N], F> + PartialDistance<N, F>,
{
}

// FIXME: inline the make_rng function in all the call sites instead! Do not just remove this comment.
fn make_rng(seed: Option<u64>) -> Pcg32 { Pcg32::seed_from_u64(seed.unwrap_or(0)) }

#[pyfunction]
fn get_rayon_parallelism() -> PyResult<usize> { Ok(rayon::current_num_threads()) }

/// Parse a distance name into a boxed dynamic distance function.
///
/// Accepted names (case-insensitive):
/// euclidean / l2, sqeuclidean / squared_euclidean, manhattan / l1 / cityblock,
/// chebyshev / linf / chessboard, cosine, arccosine / angular,
/// canberra, braycurtis / bray_curtis, hellinger, clark, chi,
/// chi_squared / chisquared / chi2, jensen_shannon / jensenshannon / js,
/// jeffrey / jeffreys, histogram_intersection / intersection.
pub(super) fn parse_distance_fn<N>(
    dist: &str,
) -> PyResult<Box<dyn DistanceFunction<[N], N> + Sync + Send>>
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

#[pyfunction]
fn _compute_pairwise_distances_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: &str,
) -> PyResult<Py<PyAny>> {
    let dist_fn = parse_distance_fn::<f32>(distance)?;
    let points_vec: Vec<Vec<f32>> = data.as_array().outer_iter().map(|row| row.to_vec()).collect();
    let points: Vec<&[f32]> = points_vec.iter().map(|row| row.as_slice()).collect();
    let matrix = crate::api::compute_pairwise_dense(&points, &|a, b| {
        dist_fn.distance(a.as_ref(), b.as_ref())
    });
    Ok(PyArray2::from_owned_array(py, matrix).into_py_any(py)?)
}

#[pyfunction]
fn _compute_pairwise_distances_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: &str,
) -> PyResult<Py<PyAny>> {
    let dist_fn = parse_distance_fn::<f64>(distance)?;
    let points_vec: Vec<Vec<f64>> = data.as_array().outer_iter().map(|row| row.to_vec()).collect();
    let points: Vec<&[f64]> = points_vec.iter().map(|row| row.as_slice()).collect();
    let matrix = crate::api::compute_pairwise_dense(&points, &|a, b| {
        dist_fn.distance(a.as_ref(), b.as_ref())
    });
    Ok(PyArray2::from_owned_array(py, matrix).into_py_any(py)?)
}

#[pyfunction]
fn _compute_pairwise_distances_condensed_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: &str,
) -> PyResult<Py<PyAny>> {
    let dist_fn = parse_distance_fn::<f32>(distance)?;
    let points_vec: Vec<Vec<f32>> = data.as_array().outer_iter().map(|row| row.to_vec()).collect();
    let points: Vec<&[f32]> = points_vec.iter().map(|row| row.as_slice()).collect();
    let vector = crate::api::compute_pairwise_condensed(&points, &|a, b| {
        dist_fn.distance(a.as_ref(), b.as_ref())
    });
    Ok(PyArray1::from_vec(py, vector).into_py_any(py)?)
}

#[pyfunction]
fn _compute_pairwise_distances_condensed_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: &str,
) -> PyResult<Py<PyAny>> {
    let dist_fn = parse_distance_fn::<f64>(distance)?;
    let points_vec: Vec<Vec<f64>> = data.as_array().outer_iter().map(|row| row.to_vec()).collect();
    let points: Vec<&[f64]> = points_vec.iter().map(|row| row.as_slice()).collect();
    let vector = crate::api::compute_pairwise_condensed(&points, &|a, b| {
        dist_fn.distance(a.as_ref(), b.as_ref())
    });
    Ok(PyArray1::from_vec(py, vector).into_py_any(py)?)
}

pub(super) fn parse_kd_distance_fn<N>(
    dist: &str,
) -> PyResult<Box<dyn KdDistanceFunction<N, N> + Sync + Send>>
where
    N: Float,
{
    match dist.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Box::new(Euclidean)),
        "sqeuclidean" | "squared_euclidean" => Ok(Box::new(SquaredEuclidean)),
        "manhattan" | "l1" | "cityblock" => Ok(Box::new(Manhattan)),
        other => Err(PyValueError::new_err(format!(
            "KdTree does not support distance '{}'; supported: euclidean, sqeuclidean, manhattan",
            other
        ))),
    }
}

mod dbscan;
mod em;
mod evaluation;
mod hdbscan;
mod hierarchical;
mod kmeans;
mod kmedoids;
mod outlier;
mod search;
mod sparse;
mod spherical;

#[pymodule]
#[pyo3(module = "fuel", name = "_fuel")]
fn _fuel<'py>(_py: Python<'py>, m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(get_rayon_parallelism))?;
    dbscan::register(m)?;
    kmeans::register(m)?;
    spherical::register(m)?;
    em::register(m)?;
    evaluation::register(m)?;
    hdbscan::register(m)?;
    hierarchical::register(m)?;
    outlier::register(m)?;
    search::register(m)?;
    m.add(
        "_compute_pairwise_distances_f32",
        wrap_pyfunction!(_compute_pairwise_distances_f32, m)?,
    )?;
    m.add(
        "_compute_pairwise_distances_f64",
        wrap_pyfunction!(_compute_pairwise_distances_f64, m)?,
    )?;
    m.add(
        "_compute_pairwise_distances_condensed_f32",
        wrap_pyfunction!(_compute_pairwise_distances_condensed_f32, m)?,
    )?;
    m.add(
        "_compute_pairwise_distances_condensed_f64",
        wrap_pyfunction!(_compute_pairwise_distances_condensed_f64, m)?,
    )?;
    kmedoids::register(m)?;
    Ok(())
}
