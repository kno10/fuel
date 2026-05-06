use ndarray::ArrayView2;
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::SeedableRng;
use rand_pcg::Pcg32;

use crate::api::query::CoordinateQuery;
use crate::api::search::{linear_scan_knn, linear_scan_range};
use crate::distance::{DistanceFunction, Euclidean};
use crate::search::covertree::{CoverTree, expansion_heuristic_from_id};
use crate::search::kdtree::{KdTree, MaxVarianceSplit};
use crate::search::linear_scan::LinearScanSearcher as RustLinearScanSearcher;
use crate::search::vptree::VPTree;
use crate::{
    DistPair, DistanceData, DistanceSearch, Float, KnnSearch, NdArrayDatasetWithDistance, ParMap,
    RangeSearch, VectorData,
};

type DistanceFn<N> = Box<dyn DistanceFunction<[N], N> + Send + Sync>;
type KdTreeDistanceFn<N> = Box<dyn super::KdDistanceFunction<N, N> + Send + Sync>;

/// Query for a non-Euclidean distance on coordinate data.
///
/// Uses two lifetimes to avoid invariance: `'data` for the dataset and distance function
/// (long-lived, from the outer scope), `'coords` for the per-row query coordinates
/// (short-lived, per iteration).
struct ExternalQuery<'data, 'coords, N>
where
    N: Float,
{
    data: &'data dyn VectorData<N>,
    coords: &'coords [N],
    distance_fn: &'data (dyn DistanceFunction<[N], N> + Sync),
}

impl<'data, 'coords, N> ExternalQuery<'data, 'coords, N>
where
    N: Float,
{
    fn new(
        data: &'data dyn VectorData<N>, coords: &'coords [N],
        distance_fn: &'data (dyn DistanceFunction<[N], N> + Sync),
    ) -> Self {
        Self { data, coords, distance_fn }
    }
}

impl<N> DistanceSearch<N> for ExternalQuery<'_, '_, N>
where
    N: Float,
{
    fn query_distance(&self, b: usize) -> N {
        self.distance_fn.distance(self.coords, self.data.point(b))
    }
}

#[pyclass]
pub struct SearchIndex {
    inner: SearchIndexInner,
}

enum SearchIndexInner {
    LinearScanF32(LinearScanSearcherF32),
    LinearScanF64(LinearScanSearcherF64),
    VpTreeF32(VPTree<f32>),
    VpTreeF64(VPTree<f64>),
    CoverTreeF32(CoverTree<f32>),
    CoverTreeF64(CoverTree<f64>),
    KdTreeF32(KdTree<f32>),
    KdTreeF64(KdTree<f64>),
}

enum LinearScanDistanceFn<N> {
    Euclidean,
    Other(DistanceFn<N>),
}

/// Run a linear-scan kNN over all rows of `queries`.
fn linear_scan_knn_batch<N, D, Q, F>(
    dataset: &D, queries: ArrayView2<N>, k: usize, exclude_self: bool, make_query: F,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
    D: crate::DistanceData<N> + Sync,
    Q: DistanceSearch<N>,
    F: Fn(usize, &[N]) -> Q + Sync,
{
    let searcher = RustLinearScanSearcher::new(dataset);
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let q = make_query(i, coords);
        let mut results = searcher.search_knn(&q, k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

/// Run a linear-scan range search over all rows of `queries`.
fn linear_scan_range_batch<N, D, Q, F>(
    dataset: &D, queries: ArrayView2<N>, radius: N, exclude_self: bool, make_query: F,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
    D: crate::DistanceData<N> + Sync,
    Q: DistanceSearch<N>,
    F: Fn(usize, &[N]) -> Q + Sync,
{
    let searcher = RustLinearScanSearcher::new(dataset);
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let q = make_query(i, coords);
        let mut results = searcher.search_range(&q, radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

struct LinearScanSearcherF32 {
    dist_fn: LinearScanDistanceFn<f32>,
}

struct LinearScanSearcherF64 {
    dist_fn: LinearScanDistanceFn<f64>,
}

impl LinearScanSearcherF32 {
    fn query_search_knn(
        &self, py: Python<'_>, data: Py<PyAny>, query: ArrayView2<f32>, k: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
        let data_view = data.as_array();
        let rows = query.nrows();
        let all_results = match &self.dist_fn {
            LinearScanDistanceFn::Euclidean => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, Euclidean);
                linear_scan_knn_batch(&dataset, query, k, exclude_self, |_i, coords| {
                    dataset.query().with_coordinates(coords)
                })
            }
            LinearScanDistanceFn::Other(dist_fn) => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &**dist_fn);
                (0..rows).par_map(|i| {
                    let query_row = query.row(i);
                    let coords = query_row.as_slice().expect("query rows must be contiguous");
                    let q = ExternalQuery::new(&dataset, coords, &**dist_fn);
                    let mut results = linear_scan_knn(&dataset, &q, k);
                    if exclude_self {
                        results.retain(|pair| pair.index != i);
                    }
                    results
                })
            }
        };
        knn_results_to_arrays(py, all_results, rows, k)
    }

    fn query_search_range(
        &self, py: Python<'_>, data: Py<PyAny>, query: ArrayView2<f32>, radius: f32,
        exclude_self: bool,
    ) -> PyResult<Vec<Vec<DistPair<f32>>>> {
        let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
        let data_view = data.as_array();
        let rows = query.nrows();
        let results = match &self.dist_fn {
            LinearScanDistanceFn::Euclidean => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, Euclidean);
                linear_scan_range_batch(&dataset, query, radius, exclude_self, |_i, coords| {
                    dataset.query().with_coordinates(coords)
                })
            }
            LinearScanDistanceFn::Other(dist_fn) => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &**dist_fn);
                (0..rows).par_map(|i| {
                    let query_row = query.row(i);
                    let coords = query_row.as_slice().expect("query rows must be contiguous");
                    let q = ExternalQuery::new(&dataset, coords, &**dist_fn);
                    let mut results = linear_scan_range(&dataset, &q, radius);
                    if exclude_self {
                        results.retain(|pair| pair.index != i);
                    }
                    results
                })
            }
        };
        Ok(results)
    }
}

impl LinearScanSearcherF64 {
    fn query_search_knn(
        &self, py: Python<'_>, data: Py<PyAny>, query: ArrayView2<f64>, k: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
        let data_view = data.as_array();
        let rows = query.nrows();
        let all_results = match &self.dist_fn {
            LinearScanDistanceFn::Euclidean => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, Euclidean);
                linear_scan_knn_batch(&dataset, query, k, exclude_self, |_i, coords| {
                    dataset.query().with_coordinates(coords)
                })
            }
            LinearScanDistanceFn::Other(dist_fn) => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &**dist_fn);
                (0..rows).par_map(|i| {
                    let query_row = query.row(i);
                    let coords = query_row.as_slice().expect("query rows must be contiguous");
                    let q = ExternalQuery::new(&dataset, coords, &**dist_fn);
                    let mut results = linear_scan_knn(&dataset, &q, k);
                    if exclude_self {
                        results.retain(|pair| pair.index != i);
                    }
                    results
                })
            }
        };
        knn_results_to_arrays(py, all_results, rows, k)
    }

    fn query_search_range(
        &self, py: Python<'_>, data: Py<PyAny>, query: ArrayView2<f64>, radius: f64,
        exclude_self: bool,
    ) -> PyResult<Vec<Vec<DistPair<f64>>>> {
        let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
        let data_view = data.as_array();
        let rows = query.nrows();
        let results = match &self.dist_fn {
            LinearScanDistanceFn::Euclidean => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, Euclidean);
                linear_scan_range_batch(&dataset, query, radius, exclude_self, |_i, coords| {
                    dataset.query().with_coordinates(coords)
                })
            }
            LinearScanDistanceFn::Other(dist_fn) => {
                let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &**dist_fn);
                (0..rows).par_map(|i| {
                    let query_row = query.row(i);
                    let coords = query_row.as_slice().expect("query rows must be contiguous");
                    let q = ExternalQuery::new(&dataset, coords, &**dist_fn);
                    let mut results = linear_scan_range(&dataset, &q, radius);
                    if exclude_self {
                        results.retain(|pair| pair.index != i);
                    }
                    results
                })
            }
        };
        Ok(results)
    }
}

#[pymethods]
impl SearchIndex {
    fn knn<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, k: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SearchIndexInner::LinearScanF32(inner) => {
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                inner.query_search_knn(py, data, query.as_array(), k, exclude_self)
            }
            SearchIndexInner::LinearScanF64(inner) => {
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                inner.query_search_knn(py, data, query.as_array(), k, exclude_self)
            }
            SearchIndexInner::VpTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = vp_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            SearchIndexInner::VpTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = vp_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            SearchIndexInner::CoverTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results =
                    cover_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            SearchIndexInner::CoverTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results =
                    cover_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            SearchIndexInner::KdTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_kd_distance_fn::<f32>(distance)?;
                let results = kd_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            SearchIndexInner::KdTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_kd_distance_fn::<f64>(distance)?;
                let results = kd_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
        }
    }

    fn radius_search<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, radius: f64,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SearchIndexInner::LinearScanF32(inner) => {
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let results = inner.query_search_range(
                    py,
                    data,
                    query.as_array(),
                    radius as f32,
                    exclude_self,
                )?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::LinearScanF64(inner) => {
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let results =
                    inner.query_search_range(py, data, query.as_array(), radius, exclude_self)?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::VpTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = vp_radius_search_query(
                    inner,
                    data_view,
                    query_view,
                    &dist_fn,
                    <f32 as Float>::cast(radius),
                    exclude_self,
                );
                range_results_to_py(py, results)
            }
            SearchIndexInner::VpTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = vp_radius_search_query(
                    inner,
                    data_view,
                    query_view,
                    &dist_fn,
                    <f64 as Float>::cast(radius),
                    exclude_self,
                );
                range_results_to_py(py, results)
            }
            SearchIndexInner::CoverTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = cover_radius_search_query(
                    inner,
                    data_view,
                    query_view,
                    &dist_fn,
                    <f32 as Float>::cast(radius),
                    exclude_self,
                );
                range_results_to_py(py, results)
            }
            SearchIndexInner::CoverTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = cover_radius_search_query(
                    inner,
                    data_view,
                    query_view,
                    &dist_fn,
                    <f64 as Float>::cast(radius),
                    exclude_self,
                );
                range_results_to_py(py, results)
            }
            SearchIndexInner::KdTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_kd_distance_fn::<f32>(distance)?;
                let results = kd_radius_search_query(
                    inner,
                    data_view,
                    query_view,
                    &dist_fn,
                    <f32 as Float>::cast(radius),
                    exclude_self,
                );
                range_results_to_py(py, results)
            }
            SearchIndexInner::KdTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_kd_distance_fn::<f64>(distance)?;
                let results = kd_radius_search_query(
                    inner,
                    data_view,
                    query_view,
                    &dist_fn,
                    <f64 as Float>::cast(radius),
                    exclude_self,
                );
                range_results_to_py(py, results)
            }
        }
    }
}

fn vp_knn_query<N>(
    tree: &VPTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    k: usize, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results = tree.search_knn(&query, query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

fn vp_radius_search_query<N>(
    tree: &VPTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    radius: N, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results: Vec<DistPair<N>> = Vec::new();
        tree.search_range(&query, radius, |pair| results.push(pair));
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

fn cover_knn_query<N>(
    tree: &CoverTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    k: usize, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results = tree.search_knn(&query, query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

fn cover_radius_search_query<N>(
    tree: &CoverTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    radius: N, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results = RangeSearch::search_range(tree, &query, radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

fn kd_knn_query<N>(
    tree: &KdTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>,
    distance_fn: &KdTreeDistanceFn<N>, k: usize, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let mut results = tree.search_knn(&dataset.query().with_coordinates(coords), query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

fn kd_radius_search_query<N>(
    tree: &KdTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>,
    distance_fn: &KdTreeDistanceFn<N>, radius: N, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    (0..queries.nrows()).par_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let mut results = tree.search_range(&dataset.query().with_coordinates(coords), radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    })
}

fn knn_results_to_arrays<'py, F>(
    py: Python<'py>, results: Vec<Vec<DistPair<F>>>, n: usize, k: usize,
) -> PyResult<Py<PyAny>>
where
    F: Float + Element,
{
    let mut idx_data: Vec<i64> = vec![-1i64; n * k];
    let mut dist_data: Vec<F> = vec![F::infinity(); n * k];
    for (i, row) in results.into_iter().enumerate() {
        for (j, pair) in row.into_iter().take(k).enumerate() {
            idx_data[i * k + j] = pair.index as i64;
            dist_data[i * k + j] = pair.distance;
        }
    }
    let idx_arr = ndarray::Array2::from_shape_vec((n, k), idx_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dist_arr = ndarray::Array2::from_shape_vec((n, k), dist_data)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let idx_py = PyArray2::from_owned_array(py, idx_arr);
    let dist_py = PyArray2::from_owned_array(py, dist_arr);
    Ok((idx_py, dist_py).into_pyobject(py)?.into())
}

fn range_results_to_py<'py, F>(
    py: Python<'py>, results: Vec<Vec<DistPair<F>>>,
) -> PyResult<Py<PyAny>>
where
    F: Float + Element,
{
    let idx_list = PyList::empty(py);
    let dist_list = PyList::empty(py);
    for row in results {
        let idx: Vec<i64> = row.iter().map(|p| p.index as i64).collect();
        let dist: Vec<F> = row.iter().map(|p| p.distance).collect();
        let idx_arr = PyArray1::from_vec(py, idx);
        let dist_arr = PyArray1::from_vec(py, dist);
        idx_list.append(idx_arr)?;
        dist_list.append(dist_arr)?;
    }
    Ok(PyTuple::new(py, &[idx_list, dist_list])?.into())
}

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn build_kd_tree_f32<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_kd_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let inner = KdTree::new(&dataset, MaxVarianceSplit);
    Ok(SearchIndex { inner: SearchIndexInner::KdTreeF32(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn build_kd_tree_f64<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_kd_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let inner = KdTree::new(&dataset, MaxVarianceSplit);
    Ok(SearchIndex { inner: SearchIndexInner::KdTreeF64(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_vp_tree_f32<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let inner = VPTree::new(&dataset, 5, &mut rng);
    Ok(SearchIndex { inner: SearchIndexInner::VpTreeF32(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_vp_tree_f64<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let inner = VPTree::new(&dataset, 5, &mut rng);
    Ok(SearchIndex { inner: SearchIndexInner::VpTreeF64(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_cover_tree_f32<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let expansion = expansion_heuristic_from_id(array.ncols() as f64);
    let inner = CoverTree::new_with_sampling(&dataset, expansion, 0, &mut rng);
    Ok(SearchIndex { inner: SearchIndexInner::CoverTreeF32(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_cover_tree_f64<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let expansion = expansion_heuristic_from_id(array.ncols() as f64);
    let inner = CoverTree::new_with_sampling(&dataset, expansion, 0, &mut rng);
    Ok(SearchIndex { inner: SearchIndexInner::CoverTreeF64(inner) })
}

fn parse_linear_scan_distance_fn_f32(distance: &str) -> PyResult<LinearScanDistanceFn<f32>> {
    match distance.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(LinearScanDistanceFn::Euclidean),
        other => Ok(LinearScanDistanceFn::Other(super::parse_distance_fn::<f32>(other)?)),
    }
}

fn parse_linear_scan_distance_fn_f64(distance: &str) -> PyResult<LinearScanDistanceFn<f64>> {
    match distance.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(LinearScanDistanceFn::Euclidean),
        other => Ok(LinearScanDistanceFn::Other(super::parse_distance_fn::<f64>(other)?)),
    }
}

#[pyfunction]
#[pyo3(signature = (_data, distance=None))]
fn build_linear_scan_f32<'py>(
    _py: Python<'py>, _data: Py<PyAny>, distance: Option<&str>,
) -> PyResult<SearchIndex> {
    let dist_fn = parse_linear_scan_distance_fn_f32(distance.unwrap_or("euclidean"))?;
    Ok(SearchIndex { inner: SearchIndexInner::LinearScanF32(LinearScanSearcherF32 { dist_fn }) })
}

#[pyfunction]
#[pyo3(signature = (_data, distance=None))]
fn build_linear_scan_f64<'py>(
    _py: Python<'py>, _data: Py<PyAny>, distance: Option<&str>,
) -> PyResult<SearchIndex> {
    let dist_fn = parse_linear_scan_distance_fn_f64(distance.unwrap_or("euclidean"))?;
    Ok(SearchIndex { inner: SearchIndexInner::LinearScanF64(LinearScanSearcherF64 { dist_fn }) })
}

// ---- registration ----------------------------------------------------------

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(build_vp_tree_f32))?;
    m.add_wrapped(wrap_pyfunction!(build_vp_tree_f64))?;
    m.add_wrapped(wrap_pyfunction!(build_cover_tree_f32))?;
    m.add_wrapped(wrap_pyfunction!(build_cover_tree_f64))?;
    m.add_wrapped(wrap_pyfunction!(build_kd_tree_f32))?;
    m.add_wrapped(wrap_pyfunction!(build_kd_tree_f64))?;
    m.add_wrapped(wrap_pyfunction!(build_linear_scan_f32))?;
    m.add_wrapped(wrap_pyfunction!(build_linear_scan_f64))?;
    m.add_class::<SearchIndex>()?;
    Ok(())
}
