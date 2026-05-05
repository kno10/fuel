use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::make_rng;
use crate::cluster::{dbscan, optics, parallel_dbscan};
use crate::distance::DistanceFunction;
use crate::search::vptree::VPTree;
use crate::{Float, NdArrayDatasetWithDistance};

fn build_dataset<'a, N>(
    array: &'a ndarray::ArrayView2<'a, N>, distance: Option<&str>,
) -> PyResult<
    NdArrayDatasetWithDistance<
        'a,
        N,
        ndarray::ArrayView2<'a, N>,
        Box<dyn DistanceFunction<[N], N> + Sync>,
    >,
>
where
    N: Float,
{
    let dist_fn: Box<dyn DistanceFunction<[N], N> + Sync> =
        super::parse_distance_fn(distance.unwrap_or("euclidean"))?;
    Ok(NdArrayDatasetWithDistance::with_distance(array, dist_fn))
}

fn build_vptree<D, F>(data: &D, seed: Option<u64>) -> VPTree<F>
where
    D: crate::DistanceData<F> + Sync,
    F: Float,
{
    let mut rng = make_rng(seed);
    VPTree::new(data, 5, &mut rng)
}

fn labels_to_py<'py>(py: Python<'py>, labels: Vec<isize>) -> PyResult<Py<PyAny>> {
    let v: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
    PyArray1::from_vec(py, v).into_py_any(py)
}

// ---- DBSCAN ----------------------------------------------------------------

macro_rules! dbscan_fn {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, eps, min_points, distance=None, seed=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, eps: f64, min_points: usize,
            distance: Option<&str>, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_dataset(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let eps = <$dtype as crate::Float>::cast(eps);
            let labels = dbscan::dbscan(&tree, &dataset, eps, min_points);
            labels_to_py(py, labels)
        }
    };
}

macro_rules! parallel_dbscan_fn {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, eps, min_points, distance=None, seed=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, eps: f64, min_points: usize,
            distance: Option<&str>, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_dataset(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let eps = <$dtype as crate::Float>::cast(eps);
            let labels = parallel_dbscan::parallel_dbscan(&tree, &dataset, eps, min_points);
            labels_to_py(py, labels)
        }
    };
}

dbscan_fn!(dbscan_f32, f32);
dbscan_fn!(dbscan_f64, f64);
parallel_dbscan_fn!(parallel_dbscan_f32, f32);
parallel_dbscan_fn!(parallel_dbscan_f64, f64);

// ---- OPTICS ----------------------------------------------------------------

/// Pyclass wrapping OpticsResult<f32>
#[pyclass]
struct OpticsResultF32 {
    inner: optics::OpticsResult<f32>,
}

/// Pyclass wrapping OpticsResult<f64>
#[pyclass]
struct OpticsResultF64 {
    inner: optics::OpticsResult<f64>,
}

macro_rules! optics_result_methods {
    ($name:ident, $float:ty) => {
        #[pymethods]
        impl $name {
            /// Processing order (indices into original data array).
            fn ordering<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let v: Vec<i64> = self.inner.ordering.iter().map(|&v| v as i64).collect();
                PyArray1::from_vec(py, v).into_py_any(py)
            }

            /// Reachability distances in processing order index.
            fn reachability<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let v: Vec<$float> = self.inner.reachability.clone();
                PyArray1::from_vec(py, v).into_py_any(py)
            }

            /// Core distances per point (NaN for non-core points).
            fn core_distance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let v: Vec<$float> = self
                    .inner
                    .core_distance
                    .iter()
                    .map(|opt| opt.unwrap_or(<$float>::NAN))
                    .collect();
                PyArray1::from_vec(py, v).into_py_any(py)
            }

            /// Predecessor indices per point (-1 if none).
            fn predecessor<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let v: Vec<i64> = self
                    .inner
                    .predecessor
                    .iter()
                    .map(|opt| opt.map(|v| v as i64).unwrap_or(-1))
                    .collect();
                PyArray1::from_vec(py, v).into_py_any(py)
            }

            /// DBSCAN-style cluster labels from the initial OPTICS run (-1 = noise).
            fn labels<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let v: Vec<i64> = self.inner.labels.iter().map(|&l| l as i64).collect();
                PyArray1::from_vec(py, v).into_py_any(py)
            }

            /// Extract Xi-based cluster labels from this OPTICS result.
            fn extract_xi<'py>(
                &self, py: Python<'py>, xi: f64, min_points: usize,
            ) -> PyResult<Py<PyAny>> {
                let xi = <$float as crate::Float>::cast(xi);
                let labels = optics::extract_xi_labels(&self.inner, xi, min_points);
                let v: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
                PyArray1::from_vec(py, v).into_py_any(py)
            }
        }
    };
}

optics_result_methods!(OpticsResultF32, f32);
optics_result_methods!(OpticsResultF64, f64);

#[pyfunction]
#[pyo3(signature = (data, eps, min_points, distance=None, seed=None))]
fn optics_f32<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f32>, eps: f64, min_points: usize,
    distance: Option<&str>, seed: Option<u64>,
) -> PyResult<OpticsResultF32> {
    let array = data.as_array();
    let dataset = build_dataset(&array, distance)?;
    let tree = build_vptree(&dataset, seed);
    let eps = f32::cast(eps);
    let inner = optics::optics(&tree, &dataset, eps, min_points);
    Ok(OpticsResultF32 { inner })
}

#[pyfunction]
#[pyo3(signature = (data, eps, min_points, distance=None, seed=None))]
fn optics_f64<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f64>, eps: f64, min_points: usize,
    distance: Option<&str>, seed: Option<u64>,
) -> PyResult<OpticsResultF64> {
    let array = data.as_array();
    let dataset = build_dataset(&array, distance)?;
    let tree = build_vptree(&dataset, seed);
    let inner = optics::optics(&tree, &dataset, eps, min_points);
    Ok(OpticsResultF64 { inner })
}

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(dbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(dbscan_f64))?;
    m.add_wrapped(wrap_pyfunction!(parallel_dbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(parallel_dbscan_f64))?;
    m.add_wrapped(wrap_pyfunction!(optics_f32))?;
    m.add_wrapped(wrap_pyfunction!(optics_f64))?;
    m.add_class::<OpticsResultF32>()?;
    m.add_class::<OpticsResultF64>()?;
    Ok(())
}
