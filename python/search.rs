use std::sync::Arc;

use ndarray::{Array2, ArrayView2};
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rand::SeedableRng;
use rand_pcg::Pcg32;

use crate::api::query::{CoordinateQuery, IndexQuery};
use crate::api::search::{PrioritySearcher, PrioritySearcherFactory};
use crate::distance::DistanceFunction;
use crate::python::KdDistanceFunction;
use crate::search::covertree::{CoverTree, expansion_heuristic_from_id};
use crate::search::kdtree::{KdTree, MaxVarianceSplit};
use crate::search::linear_scan::LinearScanSearcher as RustLinearScanSearcher;
use crate::search::precomputed::PrecomputedKnnSearcher;
use crate::search::vptree::VPTree;
use crate::{
    DistPair, DistanceData, DistanceSearch, Float, KnnSearch, NdArrayDatasetWithDistance, ParMap,
    RangeSearch, VectorData,
};

type DistanceFn<N> = Box<dyn DistanceFunction<[N], N> + Send + Sync>;
type KdTreeDistanceFn<N> = Box<dyn super::KdDistanceFunction<N, N> + Send + Sync>;

/// Query for a non-Euclidean distance on coordinate data.
///
/// Uses a long-lived `'data` lifetime for the dataset and distance function.
/// Query coordinates are owned per query to avoid borrowing issues across threads.
struct ExternalQuery<'data, N>
where
    N: Float,
{
    data: &'data (dyn VectorData<N> + Sync),
    coords: Vec<N>,
    distance_fn: &'data (dyn DistanceFunction<[N], N> + Sync),
    index: usize,
}

impl<'data, N> ExternalQuery<'data, N>
where
    N: Float,
{
    fn new(
        data: &'data (dyn VectorData<N> + Sync), coords: &[N],
        distance_fn: &'data (dyn DistanceFunction<[N], N> + Sync),
    ) -> Self {
        Self { data, coords: coords.to_vec(), distance_fn, index: 0 }
    }
}

impl<N> DistanceSearch<N> for ExternalQuery<'_, N>
where
    N: Float,
{
    fn query_distance(&self, b: usize) -> N {
        self.distance_fn.distance(self.coords.as_slice(), self.data.point(b))
    }
}

impl<N> crate::api::query::IndexQuery<N> for ExternalQuery<'_, N>
where
    N: Float,
{
    fn set_index(&mut self, idx: usize) { self.index = idx; }

    fn query_index(&self) -> usize { self.index }
}

#[pyclass]
pub struct SearchIndex {
    inner: SearchIndexInner,
}

pub(crate) enum SearchIndexInner {
    LinearScanF32(LinearScanSearcher<f32>),
    LinearScanF64(LinearScanSearcher<f64>),
    PrecomputedLinearScanF32(PrecomputedKnnSearcher<f32, PythonLinearScanSource<f32>>),
    PrecomputedLinearScanF64(PrecomputedKnnSearcher<f64, PythonLinearScanSource<f64>>),
    VpTreeF32(VPTree<f32>),
    VpTreeF64(VPTree<f64>),
    PrecomputedVpTreeF32(PrecomputedKnnSearcher<f32, VPTree<f32>>),
    PrecomputedVpTreeF64(PrecomputedKnnSearcher<f64, VPTree<f64>>),
    CoverTreeF32(CoverTree<f32>),
    CoverTreeF64(CoverTree<f64>),
    PrecomputedCoverTreeF32(PrecomputedKnnSearcher<f32, CoverTree<f32>>),
    PrecomputedCoverTreeF64(PrecomputedKnnSearcher<f64, CoverTree<f64>>),
    KdTreeF32(KdTree<f32>, Array2<f32>, KdTreeDistanceFn<f32>),
    KdTreeF64(KdTree<f64>, Array2<f64>, KdTreeDistanceFn<f64>),
    PrecomputedKdTreeF32(PrecomputedKnnSearcher<f32, KdTree<f32>>, Array2<f32>, KdTreeDistanceFn<f32>),
    PrecomputedKdTreeF64(PrecomputedKnnSearcher<f64, KdTree<f64>>, Array2<f64>, KdTreeDistanceFn<f64>),
}

// SearchIndexInner is only ever shared immutably between threads.
// All variants perform read-only search operations and do not mutate Python-owned data.
unsafe impl Sync for SearchIndexInner {}

impl SearchIndex {
    pub(crate) fn inner(&self) -> &SearchIndexInner { &self.inner }
}

impl<Q> PrioritySearcherFactory<f32, Q> for SearchIndexInner
where
    Q: DistanceSearch<f32> + Sized,
{
    type Searcher<'a>
        = Box<dyn PrioritySearcher<f32, Q> + 'a>
    where
        Self: 'a,
        Q: 'a,
        f32: 'a;

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a,
    {
        match self {
            SearchIndexInner::VpTreeF32(inner) => {
                Box::new(<VPTree<f32> as PrioritySearcherFactory<f32, Q>>::priority_searcher(inner))
            }
            SearchIndexInner::PrecomputedVpTreeF32(inner) => {
                Box::new(<PrecomputedKnnSearcher<f32, VPTree<f32>> as PrioritySearcherFactory<
                    f32,
                    Q,
                >>::priority_searcher(inner))
            }
            SearchIndexInner::CoverTreeF32(inner) => Box::new(
                <CoverTree<f32> as PrioritySearcherFactory<f32, Q>>::priority_searcher(inner),
            ),
            SearchIndexInner::PrecomputedCoverTreeF32(inner) => {
                Box::new(<PrecomputedKnnSearcher<f32, CoverTree<f32>> as PrioritySearcherFactory<
                    f32,
                    Q,
                >>::priority_searcher(inner))
            }
            SearchIndexInner::KdTreeF32(..) | SearchIndexInner::PrecomputedKdTreeF32(..) => panic!(
                "KdTree does not support arbitrary distance metrics for priority search; use VPTree or CoverTree"
            ),
            _ => unreachable!("priority_searcher called with f64 variant on f32 impl"),
        }
    }
}

impl<Q> PrioritySearcherFactory<f64, Q> for SearchIndexInner
where
    Q: DistanceSearch<f64> + Sized,
{
    type Searcher<'a>
        = Box<dyn PrioritySearcher<f64, Q> + 'a>
    where
        Self: 'a,
        Q: 'a,
        f64: 'a;

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a,
    {
        match self {
            SearchIndexInner::VpTreeF64(inner) => {
                Box::new(<VPTree<f64> as PrioritySearcherFactory<f64, Q>>::priority_searcher(inner))
            }
            SearchIndexInner::PrecomputedVpTreeF64(inner) => {
                Box::new(<PrecomputedKnnSearcher<f64, VPTree<f64>> as PrioritySearcherFactory<
                    f64,
                    Q,
                >>::priority_searcher(inner))
            }
            SearchIndexInner::CoverTreeF64(inner) => Box::new(
                <CoverTree<f64> as PrioritySearcherFactory<f64, Q>>::priority_searcher(inner),
            ),
            SearchIndexInner::PrecomputedCoverTreeF64(inner) => {
                Box::new(<PrecomputedKnnSearcher<f64, CoverTree<f64>> as PrioritySearcherFactory<
                    f64,
                    Q,
                >>::priority_searcher(inner))
            }
            SearchIndexInner::KdTreeF64(..) | SearchIndexInner::PrecomputedKdTreeF64(..) => panic!(
                "KdTree does not support arbitrary distance metrics for priority search; use VPTree or CoverTree"
            ),
            _ => unreachable!("priority_searcher called with f32 variant on f64 impl"),
        }
    }
}

impl<Q> KnnSearch<f32, Q> for SearchIndexInner
where
    Q: DistanceSearch<f32> + crate::api::query::IndexQuery<f32> + Send + ?Sized,
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<f32>> {
        match self {
            SearchIndexInner::LinearScanF32(_) => unreachable!(
                "LinearScan SearchIndex cannot be used for outlier/HDBSCAN kNN; use VPTree or CoverTree"
            ),
            SearchIndexInner::PrecomputedLinearScanF32(inner) => <PrecomputedKnnSearcher<
                f32,
                PythonLinearScanSource<f32>,
            > as KnnSearch<f32, Q>>::search_knn(
                inner, query, k
            ),
            SearchIndexInner::VpTreeF32(inner) => {
                <VPTree<f32> as KnnSearch<f32, Q>>::search_knn(inner, query, k)
            }
            SearchIndexInner::PrecomputedVpTreeF32(inner) => {
                <PrecomputedKnnSearcher<f32, VPTree<f32>> as KnnSearch<f32, Q>>::search_knn(
                    inner, query, k,
                )
            }
            SearchIndexInner::CoverTreeF32(inner) => {
                <CoverTree<f32> as KnnSearch<f32, Q>>::search_knn(inner, query, k)
            }
            SearchIndexInner::PrecomputedCoverTreeF32(inner) => {
                <PrecomputedKnnSearcher<f32, CoverTree<f32>> as KnnSearch<f32, Q>>::search_knn(
                    inner, query, k,
                )
            }
            SearchIndexInner::KdTreeF32(tree, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f32, f32> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                tree.search_knn(&kd_query, k)
            }
            SearchIndexInner::PrecomputedKdTreeF32(precomputed, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f32, f32> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                precomputed.search_knn(&kd_query, k)
            }
            _ => unreachable!("search_knn called with f64 variant on f32 impl"),
        }
    }
}

impl<Q> KnnSearch<f64, Q> for SearchIndexInner
where
    Q: DistanceSearch<f64> + crate::api::query::IndexQuery<f64> + Send + ?Sized,
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<f64>> {
        match self {
            SearchIndexInner::LinearScanF64(_) => unreachable!(
                "LinearScan SearchIndex cannot be used for outlier/HDBSCAN kNN; use VPTree or CoverTree"
            ),
            SearchIndexInner::PrecomputedLinearScanF64(inner) => <PrecomputedKnnSearcher<
                f64,
                PythonLinearScanSource<f64>,
            > as KnnSearch<f64, Q>>::search_knn(
                inner, query, k
            ),
            SearchIndexInner::VpTreeF64(inner) => {
                <VPTree<f64> as KnnSearch<f64, Q>>::search_knn(inner, query, k)
            }
            SearchIndexInner::PrecomputedVpTreeF64(inner) => {
                <PrecomputedKnnSearcher<f64, VPTree<f64>> as KnnSearch<f64, Q>>::search_knn(
                    inner, query, k,
                )
            }
            SearchIndexInner::CoverTreeF64(inner) => {
                <CoverTree<f64> as KnnSearch<f64, Q>>::search_knn(inner, query, k)
            }
            SearchIndexInner::PrecomputedCoverTreeF64(inner) => {
                <PrecomputedKnnSearcher<f64, CoverTree<f64>> as KnnSearch<f64, Q>>::search_knn(
                    inner, query, k,
                )
            }
            SearchIndexInner::KdTreeF64(tree, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f64, f64> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                tree.search_knn(&kd_query, k)
            }
            SearchIndexInner::PrecomputedKdTreeF64(precomputed, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f64, f64> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                precomputed.search_knn(&kd_query, k)
            }
            _ => unreachable!("search_knn called with f32 variant on f64 impl"),
        }
    }
}

impl<Q> KnnSearch<f32, Q> for &SearchIndexInner
where
    Q: DistanceSearch<f32> + crate::api::query::IndexQuery<f32> + Send + ?Sized,
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<f32>> {
        SearchIndexInner::search_knn(&**self, query, k)
    }
}

impl<Q> KnnSearch<f64, Q> for &SearchIndexInner
where
    Q: DistanceSearch<f64> + crate::api::query::IndexQuery<f64> + Send + ?Sized,
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<f64>> {
        SearchIndexInner::search_knn(&**self, query, k)
    }
}

impl<Q> RangeSearch<f32, Q> for SearchIndexInner
where
    Q: DistanceSearch<f32> + crate::api::query::IndexQuery<f32> + Send + ?Sized,
{
    fn search_range(&self, query: &Q, radius: f32) -> Vec<DistPair<f32>> {
        match self {
            SearchIndexInner::LinearScanF32(_) => unreachable!(
                "LinearScan SearchIndex cannot be used for outlier/HDBSCAN range search; use VPTree or CoverTree"
            ),
            SearchIndexInner::PrecomputedLinearScanF32(inner) => <PrecomputedKnnSearcher<
                f32,
                PythonLinearScanSource<f32>,
            > as RangeSearch<f32, Q>>::search_range(
                inner, query, radius
            ),
            SearchIndexInner::VpTreeF32(inner) => {
                <VPTree<f32> as RangeSearch<f32, Q>>::search_range(inner, query, radius)
            }
            SearchIndexInner::PrecomputedVpTreeF32(inner) => {
                <PrecomputedKnnSearcher<f32, VPTree<f32>> as RangeSearch<f32, Q>>::search_range(
                    inner, query, radius,
                )
            }
            SearchIndexInner::CoverTreeF32(inner) => {
                <CoverTree<f32> as RangeSearch<f32, Q>>::search_range(inner, query, radius)
            }
            SearchIndexInner::PrecomputedCoverTreeF32(inner) => {
                <PrecomputedKnnSearcher<f32, CoverTree<f32>> as RangeSearch<f32, Q>>::search_range(
                    inner, query, radius,
                )
            }
            SearchIndexInner::KdTreeF32(tree, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f32, f32> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                tree.search_range(&kd_query, radius)
            }
            SearchIndexInner::PrecomputedKdTreeF32(precomputed, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f32, f32> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                precomputed.search_range(&kd_query, radius)
            }
            _ => unreachable!("search_range called with f64 variant on f32 impl"),
        }
    }
}

impl<Q> RangeSearch<f64, Q> for SearchIndexInner
where
    Q: DistanceSearch<f64> + crate::api::query::IndexQuery<f64> + Send + ?Sized,
{
    fn search_range(&self, query: &Q, radius: f64) -> Vec<DistPair<f64>> {
        match self {
            SearchIndexInner::LinearScanF64(_) => unreachable!(
                "LinearScan SearchIndex cannot be used for outlier/HDBSCAN range search; use VPTree or CoverTree"
            ),
            SearchIndexInner::PrecomputedLinearScanF64(inner) => <PrecomputedKnnSearcher<
                f64,
                PythonLinearScanSource<f64>,
            > as RangeSearch<f64, Q>>::search_range(
                inner, query, radius
            ),
            SearchIndexInner::VpTreeF64(inner) => {
                <VPTree<f64> as RangeSearch<f64, Q>>::search_range(inner, query, radius)
            }
            SearchIndexInner::PrecomputedVpTreeF64(inner) => {
                <PrecomputedKnnSearcher<f64, VPTree<f64>> as RangeSearch<f64, Q>>::search_range(
                    inner, query, radius,
                )
            }
            SearchIndexInner::CoverTreeF64(inner) => {
                <CoverTree<f64> as RangeSearch<f64, Q>>::search_range(inner, query, radius)
            }
            SearchIndexInner::PrecomputedCoverTreeF64(inner) => {
                <PrecomputedKnnSearcher<f64, CoverTree<f64>> as RangeSearch<f64, Q>>::search_range(
                    inner, query, radius,
                )
            }
            SearchIndexInner::KdTreeF64(tree, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f64, f64> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                tree.search_range(&kd_query, radius)
            }
            SearchIndexInner::PrecomputedKdTreeF64(precomputed, data, dist_fn) => {
                let idx = query.query_index();
                let dist_fn_ref: &(dyn KdDistanceFunction<f64, f64> + Sync) = &**dist_fn;
                let dataset = NdArrayDatasetWithDistance::with_distance(data, dist_fn_ref);
                let mut kd_query = dataset.query();
                kd_query.set_index(idx);
                precomputed.search_range(&kd_query, radius)
            }
            _ => unreachable!("search_range called with f32 variant on f64 impl"),
        }
    }
}

type LinearScanDistanceFn<N> = Arc<dyn DistanceFunction<[N], N> + Send + Sync>;

pub(crate) struct PythonLinearScanSource<N>
where
    N: Float,
{
    data: Py<PyAny>,
    dist_fn: LinearScanDistanceFn<N>,
}

impl<N> PythonLinearScanSource<N> where N: Float + Element {}

impl<N, Q> KnnSearch<N, Q> for PythonLinearScanSource<N>
where
    N: Float + Element,
    Q: DistanceSearch<N> + ?Sized,
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<N>> {
        Python::try_attach(|py| {
            let pydata = self
                .data
                .as_ref()
                .extract::<PyReadonlyArray2<'_, N>>(py)
                .expect("invalid Python data for linear scan source");
            let array = pydata.as_array();
            let distance_fn: &(dyn DistanceFunction<[N], N> + Sync) = &*self.dist_fn;
            let dataset = NdArrayDatasetWithDistance::with_distance(&array, distance_fn);
            let searcher = RustLinearScanSearcher::new(&dataset);
            searcher.search_knn(query, k)
        })
        .expect("failed to attach to Python interpreter")
    }
}

impl<N, Q> RangeSearch<N, Q> for PythonLinearScanSource<N>
where
    N: Float + Element,
    Q: DistanceSearch<N> + ?Sized,
{
    fn search_range(&self, query: &Q, radius: N) -> Vec<DistPair<N>> {
        Python::try_attach(|py| {
            let pydata = self
                .data
                .as_ref()
                .extract::<PyReadonlyArray2<'_, N>>(py)
                .expect("invalid Python data for linear scan source");
            let array = pydata.as_array();
            let distance_fn: &(dyn DistanceFunction<[N], N> + Sync) = &*self.dist_fn;
            let dataset = NdArrayDatasetWithDistance::with_distance(&array, distance_fn);
            let searcher = RustLinearScanSearcher::new(&dataset);
            searcher.search_range(query, radius)
        })
        .expect("failed to attach to Python interpreter")
    }
}

/// Run a linear-scan kNN over all rows of `queries`.
fn linear_scan_knn_batch<N, D, Q, F>(
    dataset: &D, queries: ArrayView2<N>, k: usize, exclude_self: bool, make_query: F,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
    D: crate::DistanceData<N> + Sync,
    Q: DistanceSearch<N>,
    F: Fn(usize, &[N]) -> Q + Sync,
{
    let searcher = RustLinearScanSearcher::new(dataset);
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let q = make_query(i, coords);
        let mut results = searcher.search_knn(&q, k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

/// Run a linear-scan range search over all rows of `queries`.
fn linear_scan_range_batch<N, D, Q, F>(
    dataset: &D, queries: ArrayView2<N>, radius: N, exclude_self: bool, make_query: F,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
    D: crate::DistanceData<N> + Sync,
    Q: DistanceSearch<N>,
    F: Fn(usize, &[N]) -> Q + Sync,
{
    let searcher = RustLinearScanSearcher::new(dataset);
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let q = make_query(i, coords);
        let mut results = searcher.search_range(&q, radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

pub(crate) struct LinearScanSearcher<N: Float> {
    dist_fn: LinearScanDistanceFn<N>,
}

impl<N> LinearScanSearcher<N>
where
    N: Float + Element,
{
    fn query_search_knn(
        &self, py: Python<'_>, data: Py<PyAny>, query: ArrayView2<N>, k: usize, exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        let pydata = data.extract::<PyReadonlyArray2<'_, N>>(py)?;
        let data = pydata.as_array().to_owned();
        let query = query.to_owned();
        let rows = query.nrows();
        let dist_fn: &(dyn DistanceFunction<[N], N> + Sync) = &*self.dist_fn;
        let all_results = crate::py_interruptible(py, move || {
            let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
            linear_scan_knn_batch(&dataset, query.view(), k, exclude_self, |_i, coords| {
                ExternalQuery::new(&dataset, coords, dist_fn)
            })
        })?;
        knn_results_to_arrays(py, all_results, rows, k)
    }

    fn query_search_range(
        &self, py: Python<'_>, data: Py<PyAny>, query: ArrayView2<N>, radius: N, exclude_self: bool,
    ) -> PyResult<Vec<Vec<DistPair<N>>>> {
        let pydata = data.extract::<PyReadonlyArray2<'_, N>>(py)?;
        let data = pydata.as_array().to_owned();
        let query = query.to_owned();
        let dist_fn: &(dyn DistanceFunction<[N], N> + Sync) = &*self.dist_fn;
        let results = crate::py_interruptible(py, move || {
            let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
            linear_scan_range_batch(&dataset, query.view(), radius, exclude_self, |_i, coords| {
                ExternalQuery::new(&dataset, coords, dist_fn)
            })
        })?;
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
                let query_py = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query_py.as_array();
                inner.query_search_knn(py, data, query, k, exclude_self)
            }
            SearchIndexInner::PrecomputedLinearScanF32(inner) => {
                let data_py = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data = data_py.as_array();
                let query_py = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query_py.as_array();
                let dist_fn_box = super::parse_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f32], f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query,
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::LinearScanF64(inner) => {
                let query_py = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query_py.as_array();
                inner.query_search_knn(py, data, query, k, exclude_self)
            }
            SearchIndexInner::PrecomputedLinearScanF64(inner) => {
                let data_py = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data = data_py.as_array();
                let query_py = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query_py.as_array();
                let dist_fn_box = super::parse_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f64], f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query,
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::VpTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = crate::py_interruptible(py, || {
                    vp_knn_query(inner, data.view(), query.view(), &dist_fn, k, exclude_self)
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::PrecomputedVpTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f32], f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::VpTreeF64(inner) => {
                let data_py = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data = data_py.as_array();
                let query_py = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query_py.as_array();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = crate::py_interruptible(py, || {
                    vp_knn_query(inner, data, query, &dist_fn, k, exclude_self)
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::PrecomputedVpTreeF64(inner) => {
                let data_py = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let data = data_py.as_array();
                let query_py = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query_py.as_array();
                let dist_fn_box = super::parse_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f64], f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query,
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::CoverTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = crate::py_interruptible(py, || {
                    cover_knn_query(inner, data.view(), query.view(), &dist_fn, k, exclude_self)
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::PrecomputedCoverTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f32], f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::CoverTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = crate::py_interruptible(py, || {
                    cover_knn_query(inner, data.view(), query.view(), &dist_fn, k, exclude_self)
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::PrecomputedCoverTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f64], f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::KdTreeF32(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_kd_distance_fn::<f32>(distance)?;
                let results = crate::py_interruptible(py, || {
                    kd_knn_query(inner, data.view(), query.view(), &dist_fn, k, exclude_self)
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::PrecomputedKdTreeF32(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_kd_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn KdDistanceFunction<f32, f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query.view(),
                        |_, coords| dataset.query().with_coordinates(coords),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::KdTreeF64(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_kd_distance_fn::<f64>(distance)?;
                let results = crate::py_interruptible(py, || {
                    kd_knn_query(inner, data.view(), query.view(), &dist_fn, k, exclude_self)
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
            SearchIndexInner::PrecomputedKdTreeF64(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_kd_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn KdDistanceFunction<f64, f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_knn_with_query(
                        inner,
                        query.view(),
                        |_, coords| dataset.query().with_coordinates(coords),
                        k,
                        exclude_self,
                    )
                })?;
                knn_results_to_arrays(py, results, query.nrows(), k)
            }
        }
    }

    fn radius_search<'py>(
        &self, py: Python<'py>, data: Py<PyAny>, query: Py<PyAny>, distance: &str, radius: f64,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        match &self.inner {
            SearchIndexInner::LinearScanF32(inner) => {
                let query_py = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query_py.as_array();
                let results =
                    inner.query_search_range(py, data, query, radius as f32, exclude_self)?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedLinearScanF32(inner) => {
                let data_py = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let data = data_py.as_array();
                let query_py = query.extract::<PyReadonlyArray2<'_, f32>>(py)?;
                let query = query_py.as_array();
                let dist_fn_box = super::parse_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f32], f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query,
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        <f32 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::LinearScanF64(inner) => {
                let query_py = query.extract::<PyReadonlyArray2<'_, f64>>(py)?;
                let query = query_py.as_array();
                let results = inner.query_search_range(py, data, query, radius, exclude_self)?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedLinearScanF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f64], f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        <f64 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::VpTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = crate::py_interruptible(py, || {
                    vp_radius_search_query(
                        inner,
                        data.view(),
                        query.view(),
                        &dist_fn,
                        <f32 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedVpTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f32], f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        <f32 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::VpTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = crate::py_interruptible(py, || {
                    vp_radius_search_query(
                        inner,
                        data.view(),
                        query.view(),
                        &dist_fn,
                        <f64 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedVpTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f64], f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        <f64 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::CoverTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_distance_fn::<f32>(distance)?;
                let results = crate::py_interruptible(py, || {
                    cover_radius_search_query(
                        inner,
                        data.view(),
                        query.view(),
                        &dist_fn,
                        <f32 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedCoverTreeF32(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f32], f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        <f32 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::CoverTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_distance_fn::<f64>(distance)?;
                let results = crate::py_interruptible(py, || {
                    cover_radius_search_query(
                        inner,
                        data.view(),
                        query.view(),
                        &dist_fn,
                        <f64 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedCoverTreeF64(inner) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn DistanceFunction<[f64], f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query.view(),
                        |_, coords| ExternalQuery::new(&dataset, coords, dist_fn),
                        <f64 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::KdTreeF32(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_kd_distance_fn::<f32>(distance)?;
                let results = crate::py_interruptible(py, || {
                    kd_radius_search_query(
                        inner,
                        data.view(),
                        query.view(),
                        &dist_fn,
                        <f32 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedKdTreeF32(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f32>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_kd_distance_fn::<f32>(distance)?;
                let dist_fn: &(dyn KdDistanceFunction<f32, f32> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query.view(),
                        |_, coords| dataset.query().with_coordinates(coords),
                        <f32 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::KdTreeF64(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn = super::parse_kd_distance_fn::<f64>(distance)?;
                let results = crate::py_interruptible(py, || {
                    kd_radius_search_query(
                        inner,
                        data.view(),
                        query.view(),
                        &dist_fn,
                        <f64 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
            SearchIndexInner::PrecomputedKdTreeF64(inner, _, _) => {
                let data = data.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let query = query.extract::<PyReadonlyArray2<'_, f64>>(py)?.as_array().to_owned();
                let dist_fn_box = super::parse_kd_distance_fn::<f64>(distance)?;
                let dist_fn: &(dyn KdDistanceFunction<f64, f64> + Sync) = &*dist_fn_box;
                let dataset = NdArrayDatasetWithDistance::with_distance(&data, dist_fn);
                let results = crate::py_interruptible(py, || {
                    batch_range_with_query(
                        inner,
                        query.view(),
                        |_, coords| dataset.query().with_coordinates(coords),
                        <f64 as Float>::cast(radius),
                        exclude_self,
                    )
                })?;
                range_results_to_py(py, results)
            }
        }
    }
}

fn vp_knn_query<N>(
    tree: &VPTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    k: usize, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results = tree.search_knn(&query, query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

fn vp_radius_search_query<N>(
    tree: &VPTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    radius: N, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results: Vec<DistPair<N>> = Vec::new();
        tree.search_range(&query, radius, |pair| results.push(pair));
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

fn cover_knn_query<N>(
    tree: &CoverTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    k: usize, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results = tree.search_knn(&query, query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

fn cover_radius_search_query<N>(
    tree: &CoverTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>, distance_fn: &DistanceFn<N>,
    radius: N, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let query = ExternalQuery::new(&dataset, coords, &**distance_fn);
        let mut results = RangeSearch::search_range(tree, &query, radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

fn kd_knn_query<N>(
    tree: &KdTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>,
    distance_fn: &KdTreeDistanceFn<N>, k: usize, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let mut results = tree.search_knn(&dataset.query().with_coordinates(coords), query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

fn kd_radius_search_query<N>(
    tree: &KdTree<N>, data: ArrayView2<N>, queries: ArrayView2<N>,
    distance_fn: &KdTreeDistanceFn<N>, radius: N, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
{
    let dataset = NdArrayDatasetWithDistance::with_distance(&data, &**distance_fn);
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let mut results = tree.search_range(&dataset.query().with_coordinates(coords), radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

fn batch_knn_with_query<N, S, Q, F>(
    searcher: &S, queries: ArrayView2<N>, make_query: F, k: usize, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
    S: KnnSearch<N, Q> + Sync,
    Q: DistanceSearch<N> + Send,
    F: Fn(usize, &[N]) -> Q + Sync,
{
    let query_k = if exclude_self { k.saturating_add(1) } else { k };
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let mut results = searcher.search_knn(&make_query(i, coords), query_k);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
    })
}

fn batch_range_with_query<N, S, Q, F>(
    searcher: &S, queries: ArrayView2<N>, make_query: F, radius: N, exclude_self: bool,
) -> Result<Vec<Vec<DistPair<N>>>, String>
where
    N: Float,
    S: RangeSearch<N, Q> + Sync,
    Q: DistanceSearch<N> + Send,
    F: Fn(usize, &[N]) -> Q + Sync,
{
    (0..queries.nrows()).par_try_map(|i| {
        let query_row = queries.row(i);
        let coords = query_row.as_slice().expect("query rows must be contiguous");
        let mut results = searcher.search_range(&make_query(i, coords), radius);
        if exclude_self {
            results.retain(|pair| pair.index != i);
        }
        Ok(results)
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
#[pyo3(signature = (data, distance=None, precompute=None))]
fn build_kd_tree_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>,
    precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_kd_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let tree = py.detach(|| KdTree::new(&dataset, MaxVarianceSplit));
    let owned = array.to_owned();
    if let Some(max_k) = precompute {
        let inner = PrecomputedKnnSearcher::new(tree, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedKdTreeF32(inner, owned, dist_fn) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::KdTreeF32(tree, owned, dist_fn) })
    }
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, precompute=None))]
fn build_kd_tree_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>,
    precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_kd_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let tree = py.detach(|| KdTree::new(&dataset, MaxVarianceSplit));
    let owned = array.to_owned();
    if let Some(max_k) = precompute {
        let inner = PrecomputedKnnSearcher::new(tree, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedKdTreeF64(inner, owned, dist_fn) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::KdTreeF64(tree, owned, dist_fn) })
    }
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None, precompute=None))]
fn build_vp_tree_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>, seed: Option<u64>,
    precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let tree = py.detach(|| VPTree::new(&dataset, 5, &mut rng));
    if let Some(max_k) = precompute {
        let inner = PrecomputedKnnSearcher::new(tree, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedVpTreeF32(inner) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::VpTreeF32(tree) })
    }
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None, precompute=None))]
fn build_vp_tree_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>, seed: Option<u64>,
    precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let tree = py.detach(|| VPTree::new(&dataset, 5, &mut rng));
    if let Some(max_k) = precompute {
        let inner = PrecomputedKnnSearcher::new(tree, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedVpTreeF64(inner) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::VpTreeF64(tree) })
    }
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None, precompute=None))]
fn build_cover_tree_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>, seed: Option<u64>,
    precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let expansion = expansion_heuristic_from_id(array.ncols() as f64);
    let tree = py.detach(|| CoverTree::new_with_sampling(&dataset, expansion, 0, &mut rng));
    if let Some(max_k) = precompute {
        let inner = PrecomputedKnnSearcher::new(tree, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedCoverTreeF32(inner) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::CoverTreeF32(tree) })
    }
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, seed=None, precompute=None))]
fn build_cover_tree_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>, seed: Option<u64>,
    precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let array = data.as_array();
    let dist_fn = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
    let mut rng = Pcg32::seed_from_u64(seed.unwrap_or(0));
    let expansion = expansion_heuristic_from_id(array.ncols() as f64);
    let tree = py.detach(|| CoverTree::new_with_sampling(&dataset, expansion, 0, &mut rng));
    if let Some(max_k) = precompute {
        let inner = PrecomputedKnnSearcher::new(tree, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedCoverTreeF64(inner) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::CoverTreeF64(tree) })
    }
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, precompute=None))]
fn build_linear_scan_f32<'py>(
    py: Python<'py>, data: Py<PyAny>, distance: Option<&str>, precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let data_ref = data.extract::<PyReadonlyArray2<'_, f32>>(py)?;
    let array = data_ref.as_array();
    let dist_fn_box = super::parse_distance_fn::<f32>(distance.unwrap_or("euclidean"))?;
    let dist_fn: Arc<dyn DistanceFunction<[f32], f32> + Send + Sync> = Arc::from(dist_fn_box);
    if let Some(max_k) = precompute {
        let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
        let source = PythonLinearScanSource { data, dist_fn: dist_fn.clone() };
        let inner = PrecomputedKnnSearcher::new(source, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedLinearScanF32(inner) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::LinearScanF32(LinearScanSearcher { dist_fn }) })
    }
}

#[pyfunction]
#[pyo3(signature = (data, distance=None, precompute=None))]
fn build_linear_scan_f64<'py>(
    py: Python<'py>, data: Py<PyAny>, distance: Option<&str>, precompute: Option<usize>,
) -> PyResult<SearchIndex> {
    let data_ref = data.extract::<PyReadonlyArray2<'_, f64>>(py)?;
    let array = data_ref.as_array();
    let dist_fn_box = super::parse_distance_fn::<f64>(distance.unwrap_or("euclidean"))?;
    let dist_fn: Arc<dyn DistanceFunction<[f64], f64> + Send + Sync> = Arc::from(dist_fn_box);
    if let Some(max_k) = precompute {
        let dataset = NdArrayDatasetWithDistance::with_distance(&array, &*dist_fn);
        let source = PythonLinearScanSource { data, dist_fn: dist_fn.clone() };
        let inner = PrecomputedKnnSearcher::new(source, &dataset, max_k);
        Ok(SearchIndex { inner: SearchIndexInner::PrecomputedLinearScanF64(inner) })
    } else {
        Ok(SearchIndex { inner: SearchIndexInner::LinearScanF64(LinearScanSearcher { dist_fn }) })
    }
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
