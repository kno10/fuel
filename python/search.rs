use ndarray::ArrayView2;
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::SeedableRng;
use rand_pcg::Pcg32;

use crate::api::query::CoordinateQuery;
use crate::distance::DistanceFunction;
use crate::search::covertree::{CoverTree, expansion_heuristic_from_id};
use crate::search::kdtree::{KdTree, MaxVarianceSplit};
use crate::search::linear_scan::LinearScanSearcher as RustLinearScanSearcher;
use crate::search::vptree::VPTree;
use crate::{
    DistPair, DistanceData, DistanceSearch, Float, KnnSearch, NdArrayDatasetWithDistance,
    RangeSearch,
};

type DistanceFn<N> = Box<dyn DistanceFunction<[N], N> + Send + Sync>;
type KdTreeDistanceFn<N> = Box<dyn super::KdDistanceFunction<N, N> + Send + Sync>;

struct ExternalQuery<'a, N>
where
    N: Float,
{
    data: ArrayView2<'a, N>,
    coords: &'a [N],
    distance_fn: &'a dyn DistanceFunction<[N], N>,
}

impl<'a, N> ExternalQuery<'a, N>
where
    N: Float,
{
    fn new(
        data: ArrayView2<'a, N>, coords: &'a [N], distance_fn: &'a dyn DistanceFunction<[N], N>,
    ) -> Self {
        Self { data, coords, distance_fn }
    }
}

impl<'a, N> DistanceSearch<N> for ExternalQuery<'a, N>
where
    N: Float,
{
    fn query_distance(&self, b: usize) -> N {
        let data_row = self.data.row(b);
        let target = data_row.as_slice().expect("data rows must be contiguous");
        self.distance_fn.distance(self.coords, target)
    }
}

#[pyclass]
pub struct LinearScanSearcher {
    inner: LinearScanSearcherInner,
}

enum LinearScanSearcherInner {
    F32(LinearScanSearcherF32),
    F64(LinearScanSearcherF64),
}

struct LinearScanSearcherF32 {
    dist_fn: DistanceFn<f32>,
}

struct LinearScanSearcherF64 {
    dist_fn: DistanceFn<f64>,
}

impl LinearScanSearcherF32 {
    fn query_search_knn(
        &self,
        py: Python<'_>,
        data: Py<PyAny>,
        query: ArrayView2<f32>,
        k: usize,
        exclude_self: bool,
    ) -> PyResult<Vec<Vec<DistPair<f32>>>> {
        let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
        let data_view = data.as_array();
        let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &*self.dist_fn);
        let searcher = RustLinearScanSearcher::new(&dataset);
        let results = (0..query.nrows())
            .map(|i| {
                let query_row = query.row(i);
                let coords = query_row.as_slice().expect("query rows must be contiguous");
                let query = ExternalQuery::new(data_view, coords, &*self.dist_fn);
                let mut results = searcher.search_knn(&query, k);
                if exclude_self {
                    results.retain(|pair| pair.index != i);
                }
                results
            })
            .collect();
        Ok(results)
    }

    fn query_search_range(
        &self,
        py: Python<'_>,
        data: Py<PyAny>,
        query: ArrayView2<f32>,
        radius: f32,
        exclude_self: bool,
    ) -> PyResult<Vec<Vec<DistPair<f32>>>> {
        let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
        let data_view = data.as_array();
        let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &*self.dist_fn);
        let searcher = RustLinearScanSearcher::new(&dataset);
        let results = (0..query.nrows())
            .map(|i| {
                let query_row = query.row(i);
                let coords = query_row.as_slice().expect("query rows must be contiguous");
                let query = ExternalQuery::new(data_view, coords, &*self.dist_fn);
                let mut results = searcher.search_range(&query, radius);
                if exclude_self {
                    results.retain(|pair| pair.index != i);
                }
                results
            })
            .collect();
        Ok(results)
    }
}

impl LinearScanSearcherF64 {
    fn query_search_knn(
        &self,
        py: Python<'_>,
        data: Py<PyAny>,
        query: ArrayView2<f64>,
        k: usize,
        exclude_self: bool,
    ) -> PyResult<Vec<Vec<DistPair<f64>>>> {
        let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
        let data_view = data.as_array();
        let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &*self.dist_fn);
        let searcher = RustLinearScanSearcher::new(&dataset);
        let results = (0..query.nrows())
            .map(|i| {
                let query_row = query.row(i);
                let coords = query_row.as_slice().expect("query rows must be contiguous");
                let query = ExternalQuery::new(data_view, coords, &*self.dist_fn);
                let mut results = searcher.search_knn(&query, k);
                if exclude_self {
                    results.retain(|pair| pair.index != i);
                }
                results
            })
            .collect();
        Ok(results)
    }

    fn query_search_range(
        &self,
        py: Python<'_>,
        data: Py<PyAny>,
        query: ArrayView2<f64>,
        radius: f64,
        exclude_self: bool,
    ) -> PyResult<Vec<Vec<DistPair<f64>>>> {
        let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
        let data_view = data.as_array();
        let dataset = NdArrayDatasetWithDistance::with_distance(&data_view, &*self.dist_fn);
        let searcher = RustLinearScanSearcher::new(&dataset);
        let results = (0..query.nrows())
            .map(|i| {
                let query_row = query.row(i);
                let coords = query_row.as_slice().expect("query rows must be contiguous");
                let query = ExternalQuery::new(data_view, coords, &*self.dist_fn);
                let mut results = searcher.search_range(&query, radius);
                if exclude_self {
                    results.retain(|pair| pair.index != i);
                }
                results
            })
            .collect();
        Ok(results)
    }
}

#[pymethods]
impl LinearScanSearcher {
    fn knn<'py>(
        &self,
        py: Python<'py>,
        data: Py<PyAny>,
        query: Py<PyAny>,
        _distance: &str,
        k: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            LinearScanSearcherInner::F32(inner) => {
                let query = query.as_ref().extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let results = inner.query_search_knn(py, data, query.as_array(), k, exclude_self)?;
                knn_results_to_arrays(py, results, query.as_array().nrows(), k)
            }
            LinearScanSearcherInner::F64(inner) => {
                let query = query.as_ref().extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let results = inner.query_search_knn(py, data, query.as_array(), k, exclude_self)?;
                knn_results_to_arrays(py, results, query.as_array().nrows(), k)
            }
        }
    }

    fn radius_search<'py>(
        &self,
        py: Python<'py>,
        data: Py<PyAny>,
        query: Py<PyAny>,
        _distance: &str,
        radius: f64,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            LinearScanSearcherInner::F32(inner) => {
                let query = query.as_ref().extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let results = inner.query_search_range(py, data, query.as_array(), radius as f32, exclude_self)?;
                range_results_to_py(py, results)
            }
            LinearScanSearcherInner::F64(inner) => {
                let query = query.as_ref().extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let results = inner.query_search_range(py, data, query.as_array(), radius, exclude_self)?;
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
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    let do_query = |i: usize| -> Vec<DistPair<N>> {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(data, coords, &**distance_fn);
        let mut results = tree.search_knn(&query, query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..queries.nrows()).into_par_iter().map(do_query).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..queries.nrows()).map(do_query).collect()
    }
}

fn vp_radius_search_query<N>(
    tree: &VPTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    radius: N, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let do_query = |i: usize| -> Vec<DistPair<N>> {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(data, coords, &**distance_fn);
        let mut results: Vec<DistPair<N>> = Vec::new();
        tree.search_range(&query, radius, |pair| results.push(pair));
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..queries.nrows()).into_par_iter().map(do_query).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..queries.nrows()).map(do_query).collect()
    }
}

fn cover_knn_query<N>(
    tree: &CoverTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    k: usize, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    let do_query = |i: usize| -> Vec<DistPair<N>> {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(data, coords, &**distance_fn);
        let mut results = tree.search_knn(&query, query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..queries.nrows()).into_par_iter().map(do_query).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..queries.nrows()).map(do_query).collect()
    }
}

fn cover_radius_search_query<N>(
    tree: &CoverTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    radius: N, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let do_query = |i: usize| -> Vec<DistPair<N>> {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(data, coords, &**distance_fn);
        let mut results = crate::api::search::RangeSearch::search_range(tree, &query, radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..queries.nrows()).into_par_iter().map(do_query).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..queries.nrows()).map(do_query).collect()
    }
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
    let do_query = |i: usize| -> Vec<DistPair<N>> {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let mut q = dataset.query();
        q.set_coordinates(coords);
        let mut results = tree.search_knn(&q, query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..queries.nrows()).into_par_iter().map(do_query).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..queries.nrows()).map(do_query).collect()
    }
}

fn kd_radius_search_query<N>(
    tree: &KdTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>,
    distance_fn: &KdTreeDistanceFn<N>, radius: N, exclude_self: bool,
) -> Vec<Vec<DistPair<N>>>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    let do_query = |i: usize| -> Vec<DistPair<N>> {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let q = dataset.query().with_coordinates(coords);
        let mut results = crate::api::search::RangeSearch::search_range(tree, &q, radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        results
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..queries.nrows()).into_par_iter().map(do_query).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..queries.nrows()).map(do_query).collect()
    }
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

#[pyclass]
pub struct VpTreeIndex {
    inner: VpTreeIndexInner,
}

enum VpTreeIndexInner {
    F32(VPTree<f32>),
    F64(VPTree<f64>),
}

#[pyclass]
pub struct CoverTreeIndex {
    inner: CoverTreeIndexInner,
}

enum CoverTreeIndexInner {
    F32(CoverTree<f32>),
    F64(CoverTree<f64>),
}

#[pymethods]
impl VpTreeIndex {
    fn knn<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, k: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            VpTreeIndexInner::F32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = vp_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            VpTreeIndexInner::F64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = vp_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
        }
    }

    fn radius_search<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, radius: f64,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            VpTreeIndexInner::F32(inner) => {
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
            VpTreeIndexInner::F64(inner) => {
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
        }
    }
}

#[pymethods]
impl CoverTreeIndex {
    fn knn<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, k: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            CoverTreeIndexInner::F32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results =
                    cover_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            CoverTreeIndexInner::F64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results =
                    cover_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
        }
    }

    fn radius_search<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, radius: f64,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            CoverTreeIndexInner::F32(inner) => {
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
            CoverTreeIndexInner::F64(inner) => {
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
        }
    }
}

#[pyclass]
pub struct KdTreeIndex {
    inner: KdTreeIndexInner,
}

enum KdTreeIndexInner {
    F32(KdTree<f32>),
    F64(KdTree<f64>),
}

#[pymethods]
impl KdTreeIndex {
    fn knn<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, k: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            KdTreeIndexInner::F32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data_view = data.as_array();
                let query_view = query.as_array();
                let dist_fn = super::parse_kd_distance_fn::<f32>(distance)?;
                let results = kd_knn_query(inner, data_view, query_view, &dist_fn, k, exclude_self);
                knn_results_to_arrays(py, results, query_view.nrows(), k)
            }
            KdTreeIndexInner::F64(inner) => {
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
            KdTreeIndexInner::F32(inner) => {
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
            KdTreeIndexInner::F64(inner) => {
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

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn build_kd_tree_f32<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>,
) -> PyResult<KdTreeIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_kd_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let inner = KdTree::new(&dataset, MaxVarianceSplit);
    Ok(KdTreeIndex { inner: KdTreeIndexInner::F32(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn build_kd_tree_f64<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>,
) -> PyResult<KdTreeIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_kd_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let inner = KdTree::new(&dataset, MaxVarianceSplit);
    Ok(KdTreeIndex { inner: KdTreeIndexInner::F64(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_vp_tree_f32<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<VpTreeIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let inner = VPTree::new(&dataset, 5, &mut rng);
    Ok(VpTreeIndex { inner: VpTreeIndexInner::F32(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_vp_tree_f64<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<VpTreeIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let inner = VPTree::new(&dataset, 5, &mut rng);
    Ok(VpTreeIndex { inner: VpTreeIndexInner::F64(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_cover_tree_f32<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<CoverTreeIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let expansion = expansion_heuristic_from_id(array.ncols() as f64);
    let inner = CoverTree::new_with_sampling(&dataset, expansion, 0, &mut rng);
    Ok(CoverTreeIndex { inner: CoverTreeIndexInner::F32(inner) })
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None))]
fn build_cover_tree_f64<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>, seed: Option<u64>,
) -> PyResult<CoverTreeIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let expansion = expansion_heuristic_from_id(array.ncols() as f64);
    let inner = CoverTree::new_with_sampling(&dataset, expansion, 0, &mut rng);
    Ok(CoverTreeIndex { inner: CoverTreeIndexInner::F64(inner) })
}

#[pyfunction]
#[pyo3(signature = (_data, distance=None))]
fn build_linear_scan_f32<'py>(
    _py: Python<'py>, _data: Py<PyAny>, distance: Option<&str>,
) -> PyResult<LinearScanSearcher> {
    let dist_fn = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    Ok(LinearScanSearcher {
        inner: LinearScanSearcherInner::F32(LinearScanSearcherF32 { dist_fn }),
    })
}

#[pyfunction]
#[pyo3(signature = (_data, distance=None))]
fn build_linear_scan_f64<'py>(
    _py: Python<'py>, _data: Py<PyAny>, distance: Option<&str>,
) -> PyResult<LinearScanSearcher> {
    let dist_fn = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    Ok(LinearScanSearcher {
        inner: LinearScanSearcherInner::F64(LinearScanSearcherF64 { dist_fn }),
    })
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
    m.add_class::<LinearScanSearcher>()?;
    m.add_class::<VpTreeIndex>()?;
    m.add_class::<CoverTreeIndex>()?;
    m.add_class::<KdTreeIndex>()?;
    Ok(())
}
