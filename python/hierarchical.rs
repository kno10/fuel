use cluster::hierarchical::extraction::{
    cut_dendrogram_by_height, cut_dendrogram_by_number_of_clusters,
};
use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::make_rng;
use crate::cluster::hierarchical;
use crate::distance::{DistanceFunction, Euclidean};
use crate::search::vptree::VPTree;
use crate::{Float, NdArrayDatasetWithDistance, cluster};

fn to_py_array1_i64<'py, I>(py: Python<'py>, values: I) -> PyResult<Py<PyAny>>
where
    I: IntoIterator<Item = i64>,
{
    let values = values.into_iter().collect::<Vec<_>>();
    PyArray1::from_vec(py, values).into_py_any(py)
}

#[pyclass]
struct MergeHistoryF32 {
    history: hierarchical::MergeHistory<f32>,
}

#[pyclass]
struct MergeHistoryF64 {
    history: hierarchical::MergeHistory<f64>,
}

macro_rules! merge_history_methods {
    ($name:ident, $float:ty) => {
        #[pymethods]
        impl $name {
            fn __len__(&self) -> usize { self.history.len() }

            fn row<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Py<PyAny>> {
                if idx >= self.history.len() {
                    return Err(PyIndexError::new_err("row index out of range"));
                }

                let merge = self.history.get(idx).expect("index checked");
                Ok((
                    merge.idx1 as i64,
                    merge.idx2 as i64,
                    merge.distance,
                    merge.size as i64,
                    merge.prototype().map(|v| v as i64),
                )
                    .into_pyobject(py)?
                    .into())
            }

            fn idx1<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                to_py_array1_i64(py, self.history.idx1.iter().map(|&v| v as i64))
            }

            fn idx2<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                to_py_array1_i64(py, self.history.idx2.iter().map(|&v| v as i64))
            }

            fn distance<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                to_py_array1_i64(py, self.history.distance.iter().copied().map(|v| v as i64))
            }

            fn size<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                to_py_array1_i64(py, self.history.size.iter().map(|&v| v as i64))
            }

            fn prototype<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                if let Some(prototype) = &self.history.prototype {
                    to_py_array1_i64(py, prototype.iter().map(|&v| v as i64))
                } else {
                    Ok(py.None())
                }
            }

            fn to_scipy_linkage<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let n = self.history.len();
                let array = PyArray2::<$float>::zeros(py, [n, 4], false);
                let mut view = unsafe { array.as_array_mut() };
                for i in 0..n {
                    view[(i, 0)] = self.history.idx1[i] as $float;
                    view[(i, 1)] = self.history.idx2[i] as $float;
                    view[(i, 2)] = self.history.distance[i];
                    view[(i, 3)] = self.history.size[i] as $float;
                }
                Ok(array.into_pyobject(py)?.into())
            }

            fn cut_by_number_of_clusters<'py>(
                &self, py: Python<'py>, k: usize,
            ) -> PyResult<Py<PyAny>> {
                let labels = cut_dendrogram_by_number_of_clusters(&self.history, k);
                to_py_array1_i64(py, labels.iter().map(|&v| v as i64))
            }

            fn cut_by_height<'py>(
                &self, py: Python<'py>, threshold: $float,
            ) -> PyResult<Py<PyAny>> {
                let labels = cut_dendrogram_by_height(&self.history, threshold);
                to_py_array1_i64(py, labels.iter().map(|&v| v as i64))
            }
        }
    };
}

merge_history_methods!(MergeHistoryF32, f32);
merge_history_methods!(MergeHistoryF64, f64);

macro_rules! linkage_wrapper {
    ($name:ident, $algo:path, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, linkage, distance=None))]
        fn $name<'py>(
            py: Python<'py>,
            data: PyReadonlyArray2<'py, $dtype>,
            linkage: &str,
            distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_hier_dataset::<$dtype>(&array, distance)?;
            let linkage = linkage.to_ascii_lowercase();
            match linkage.as_str() {
                "single" | "complete" | "average" | "group_average" | "weighted_average"
                | "centroid" | "median" | "ward" | "missq" | "minimum_sum_squares" | "mnssq"
                | "minimum_variance_increase" | "mivar" | "minimum_variance" | "mnvar" => {}
                _ => {
                    return Err(PyValueError::new_err(
                        "unsupported linkage: expected one of single, complete, average, group_average, weighted_average, centroid, median, ward, minimum_sum_squares, minimum_variance, minimum_variance_increase",
                    ))
                }
            }
            let history = crate::py_interruptible(py, || {
                match linkage.as_str() {
                    "single" => $algo(&dataset, hierarchical::SingleLinkage),
                    "complete" => $algo(&dataset, hierarchical::CompleteLinkage),
                    "average" | "group_average" => $algo(&dataset, hierarchical::GroupAverageLinkage),
                    "weighted_average" => $algo(&dataset, hierarchical::WeightedAverageLinkage),
                    "centroid" => $algo(&dataset, hierarchical::CentroidLinkage),
                    "median" => $algo(&dataset, hierarchical::MedianLinkage),
                    "ward" | "missq" => $algo(&dataset, hierarchical::WardLinkage),
                    "minimum_sum_squares" | "mnssq" => {
                        $algo(&dataset, hierarchical::MinimumSumSquaresLinkage)
                    }
                    "minimum_variance_increase" | "mivar" => {
                        $algo(&dataset, hierarchical::MinimumVarianceIncreaseLinkage)
                    }
                    "minimum_variance" | "mnvar" => {
                        $algo(&dataset, hierarchical::MinimumVarianceLinkage)
                    }
                    _ => unreachable!("validated above"),
                }
            })?;
            Ok(Py::new(py, $wrapper { history })?.into())
        }
    };
}

macro_rules! geometric_linkage_wrapper {
    ($name:ident, $algo:path, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            data: PyReadonlyArray2<'py, $dtype>,
            linkage: &str,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDatasetWithDistance::with_distance(&array, Euclidean);
            let linkage = linkage.to_ascii_lowercase();
            match linkage.as_str() {
                "average" | "group_average" | "centroid" | "ward" | "missq"
                | "minimum_sum_squares" | "mnssq" | "minimum_variance_increase" | "mivar"
                | "minimum_variance" | "mnvar" => {}
                _ => {
                    return Err(PyValueError::new_err(
                        "unsupported geometric linkage: expected one of average, group_average, centroid, ward, minimum_sum_squares, minimum_variance, minimum_variance_increase",
                    ))
                }
            }
            let history = crate::py_interruptible(py, || {
                match linkage.as_str() {
                    "average" | "group_average" => {
                        $algo(&dataset, hierarchical::GroupAverageLinkage)
                    }
                    "centroid" => $algo(&dataset, hierarchical::CentroidLinkage),
                    "ward" | "missq" => $algo(&dataset, hierarchical::WardLinkage),
                    "minimum_sum_squares" | "mnssq" => {
                        $algo(&dataset, hierarchical::MinimumSumSquaresLinkage)
                    }
                    "minimum_variance_increase" | "mivar" => {
                        $algo(&dataset, hierarchical::MinimumVarianceIncreaseLinkage)
                    }
                    "minimum_variance" | "mnvar" => {
                        $algo(&dataset, hierarchical::MinimumVarianceLinkage)
                    }
                    _ => unreachable!("validated above"),
                }
            })?;
            Ok(Py::new(py, $wrapper { history })?.into())
        }
    };
}

linkage_wrapper!(agnes_f32, hierarchical::agnes, f32, MergeHistoryF32);
linkage_wrapper!(agnes_f64, hierarchical::agnes, f64, MergeHistoryF64);
linkage_wrapper!(anderberg_f32, hierarchical::anderberg, f32, MergeHistoryF32);
linkage_wrapper!(anderberg_f64, hierarchical::anderberg, f64, MergeHistoryF64);
linkage_wrapper!(muellner_f32, hierarchical::muellner, f32, MergeHistoryF32);
linkage_wrapper!(muellner_f64, hierarchical::muellner, f64, MergeHistoryF64);
linkage_wrapper!(nn_chain_f32, hierarchical::nn_chain, f32, MergeHistoryF32);
linkage_wrapper!(nn_chain_f64, hierarchical::nn_chain, f64, MergeHistoryF64);

geometric_linkage_wrapper!(
    geometric_nn_chain_f32,
    hierarchical::geometric_nn_chain,
    f32,
    MergeHistoryF32
);
geometric_linkage_wrapper!(
    geometric_nn_chain_f64,
    hierarchical::geometric_nn_chain,
    f64,
    MergeHistoryF64
);

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn slink_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_hier_dataset::<f32>(&array, distance)?;
    let history = crate::py_interruptible(py, || hierarchical::slink(&dataset))?;
    Ok(Py::new(py, MergeHistoryF32 { history })?.into())
}

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn slink_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_hier_dataset::<f64>(&array, distance)?;
    let history = crate::py_interruptible(py, || hierarchical::slink(&dataset))?;
    Ok(Py::new(py, MergeHistoryF64 { history })?.into())
}

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn clink_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_hier_dataset::<f32>(&array, distance)?;
    let history = crate::py_interruptible(py, || hierarchical::clink(&dataset))?;
    Ok(Py::new(py, MergeHistoryF32 { history })?.into())
}

#[pyfunction]
#[pyo3(signature = (data, distance=None))]
fn clink_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_hier_dataset::<f64>(&array, distance)?;
    let history = crate::py_interruptible(py, || hierarchical::clink(&dataset))?;
    Ok(Py::new(py, MergeHistoryF64 { history })?.into())
}

fn build_vptree<D, F>(data: &D, sample_size: usize, seed: Option<u64>) -> VPTree<F>
where
    D: crate::DistanceData<F> + Sync,
    F: Float,
{
    let mut rng = make_rng(seed);
    VPTree::new(data, sample_size.max(2), &mut rng)
}

fn build_hier_dataset<'a, N>(
    array: &'a ArrayView2<'a, N>, distance: Option<&str>,
) -> PyResult<
    NdArrayDatasetWithDistance<'a, N, ArrayView2<'a, N>, Box<dyn DistanceFunction<[N], N> + Sync>>,
>
where
    N: Float,
{
    let dist_fn: Box<dyn DistanceFunction<[N], N> + Sync> =
        super::parse_distance_fn(distance.unwrap_or("euclidean"))?;
    Ok(NdArrayDatasetWithDistance::with_distance(array, dist_fn))
}

macro_rules! search_single_link_wrapper {
    ($name:ident, $algo:path, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, sample_size, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, sample_size: usize,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_hier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, sample_size, seed);
            let history = crate::py_interruptible(py, || $algo(&tree, &dataset))?;
            Ok(Py::new(py, $wrapper { history })?.into())
        }
    };
}

macro_rules! search_single_link_wrapper_slack {
    ($name:ident, $algo:path, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, slack, sample_size, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, slack: usize, sample_size: usize,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_hier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, sample_size, seed);
            let history = crate::py_interruptible(py, || $algo(&tree, &dataset, slack))?;
            Ok(Py::new(py, $wrapper { history })?.into())
        }
    };
}

search_single_link_wrapper!(
    boruvka_searchers_single_link_f32,
    hierarchical::boruvka_searchers_single_link,
    f32,
    MergeHistoryF32
);
search_single_link_wrapper!(
    boruvka_searchers_single_link_f64,
    hierarchical::boruvka_searchers_single_link,
    f64,
    MergeHistoryF64
);
search_single_link_wrapper!(
    heap_of_searchers_single_link_f32,
    hierarchical::heap_of_searchers_single_link,
    f32,
    MergeHistoryF32
);
search_single_link_wrapper!(
    heap_of_searchers_single_link_f64,
    hierarchical::heap_of_searchers_single_link,
    f64,
    MergeHistoryF64
);
search_single_link_wrapper_slack!(
    buffered_search_single_link_f32,
    hierarchical::buffered_search_single_link,
    f32,
    MergeHistoryF32
);
search_single_link_wrapper_slack!(
    buffered_search_single_link_f64,
    hierarchical::buffered_search_single_link,
    f64,
    MergeHistoryF64
);
search_single_link_wrapper_slack!(
    lazy_buffered_search_single_link_f32,
    hierarchical::lazy_buffered_search_single_link,
    f32,
    MergeHistoryF32
);
search_single_link_wrapper_slack!(
    lazy_buffered_search_single_link_f64,
    hierarchical::lazy_buffered_search_single_link,
    f64,
    MergeHistoryF64
);
search_single_link_wrapper!(
    restarting_search_single_link_f32,
    hierarchical::restarting_search_single_link,
    f32,
    MergeHistoryF32
);
search_single_link_wrapper!(
    restarting_search_single_link_f64,
    hierarchical::restarting_search_single_link,
    f64,
    MergeHistoryF64
);

#[pyfunction]
fn incremental_nn_chain_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, linkage: &str, sample_size: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, Euclidean);
    let tree = build_vptree(&dataset, sample_size, seed);
    let linkage = linkage.to_ascii_lowercase();
    match linkage.as_str() {
        "average" | "group_average" | "centroid" | "ward" | "missq" | "minimum_sum_squares"
        | "mnssq" | "minimum_variance_increase" | "mivar" | "minimum_variance" | "mnvar" => {}
        _ => {
            return Err(PyValueError::new_err(
                "unsupported incremental linkage: expected one of average, group_average, centroid, ward, minimum_sum_squares, minimum_variance, minimum_variance_increase",
            ));
        }
    }
    let history = crate::py_interruptible(py, || {
            match linkage.as_str() {
                "average" | "group_average" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::GroupAverageLinkage,
                ),
                "centroid" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::CentroidLinkage,
                ),
                "ward" | "missq" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::WardLinkage,
                ),
                "minimum_sum_squares" | "mnssq" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::MinimumSumSquaresLinkage,
                ),
                "minimum_variance_increase" | "mivar" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::MinimumVarianceIncreaseLinkage,
                ),
                "minimum_variance" | "mnvar" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::MinimumVarianceLinkage,
                ),
                _ => unreachable!("validated above"),
            }
        })?;
    Ok(Py::new(py, MergeHistoryF32 { history })?.into())
}

#[pyfunction]
fn incremental_nn_chain_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, linkage: &str, sample_size: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::with_distance(&array, Euclidean);
    let tree = build_vptree(&dataset, sample_size, seed);
    let linkage = linkage.to_ascii_lowercase();
    match linkage.as_str() {
        "average" | "group_average" | "centroid" | "ward" | "missq" | "minimum_sum_squares"
        | "mnssq" | "minimum_variance_increase" | "mivar" | "minimum_variance" | "mnvar" => {}
        _ => {
            return Err(PyValueError::new_err(
                "unsupported incremental linkage: expected one of average, group_average, centroid, ward, minimum_sum_squares, minimum_variance, minimum_variance_increase",
            ));
        }
    }
    let history = crate::py_interruptible(py, || {
            match linkage.as_str() {
                "average" | "group_average" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::GroupAverageLinkage,
                ),
                "centroid" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::CentroidLinkage,
                ),
                "ward" | "missq" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::WardLinkage,
                ),
                "minimum_sum_squares" | "mnssq" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::MinimumSumSquaresLinkage,
                ),
                "minimum_variance" | "mnvar" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::MinimumVarianceLinkage,
                ),
                "minimum_variance_increase" | "mivar" => hierarchical::incremental_nn_chain(
                    &tree,
                    &dataset,
                    hierarchical::MinimumVarianceIncreaseLinkage,
                ),
                _ => unreachable!("validated above"),
            }
        })?;
    Ok(Py::new(py, MergeHistoryF64 { history })?.into())
}

macro_rules! set_linkage_wrapper {
    ($name:ident, $algo:ident, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, linkage, distance=None))]
        fn $name<'py>(
            py: Python<'py>,
            data: PyReadonlyArray2<'py, $dtype>,
            linkage: &str,
            distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_hier_dataset::<$dtype>(&array, distance)?;
            let linkage = linkage.to_ascii_lowercase();
            match linkage.as_str() {
                "single" | "complete" | "average" | "group_average" | "ward" | "missq"
                | "minimum_sum_squares" | "mnssq" | "minimum_variance_increase" | "mivar"
                | "minimum_variance" | "mnvar" | "minimax" | "hausdorff" | "medoid"
                | "minimum_sum" | "mnsum" | "minimum_sum_increase" | "misum" => {}
                _ => {
                    return Err(PyValueError::new_err(
                        "unsupported set linkage: expected one of single, complete, average, group_average, ward, minimum_sum_squares, minimum_variance, minimum_variance_increase, minimax, hausdorff, medoid, minimum_sum, minimum_sum_increase",
                    ))
                }
            }
            let history = crate::py_interruptible(py, || {
                    match linkage.as_str() {
                        "single" => {
                            hierarchical::$algo::<_, hierarchical::SingleLinkage, _, _>(&dataset)
                        }
                        "complete" => {
                            hierarchical::$algo::<_, hierarchical::CompleteLinkage, _, _>(&dataset)
                        }
                        "average" | "group_average" => {
                            hierarchical::$algo::<_, hierarchical::GroupAverageLinkage, _, _>(
                                &dataset,
                            )
                        }
                        "ward" | "missq" => {
                            hierarchical::$algo::<_, hierarchical::WardLinkage, _, _>(&dataset)
                        }
                        "minimum_sum_squares" | "mnssq" => {
                            hierarchical::$algo::<_, hierarchical::MinimumSumSquaresLinkage, _, _>(
                                &dataset,
                            )
                        }
                        "minimum_variance_increase" | "mivar" => {
                            hierarchical::$algo::<_, hierarchical::MinimumVarianceIncreaseLinkage, _, _>(&dataset)
                        }
                        "minimum_variance" | "mnvar" => {
                            hierarchical::$algo::<_, hierarchical::MinimumVarianceLinkage, _, _>(
                                &dataset,
                            )
                        }
                        "minimax" => {
                            hierarchical::$algo::<_, hierarchical::MinimaxLinkage, _, _>(&dataset)
                        }
                        "hausdorff" => {
                            hierarchical::$algo::<_, hierarchical::HausdorffLinkage, _, _>(&dataset)
                        }
                        "medoid" => {
                            hierarchical::$algo::<_, hierarchical::MedoidLinkage, _, _>(&dataset)
                        }
                        "minimum_sum" | "mnsum" => {
                            hierarchical::$algo::<_, hierarchical::MinimumSumLinkage, _, _>(&dataset)
                        }
                        "minimum_sum_increase" | "misum" => {
                            hierarchical::$algo::<_, hierarchical::MinimumSumIncreaseLinkage, _, _>(
                                &dataset,
                            )
                        }
                        _ => unreachable!("validated above"),
                    }
                })?;
            Ok(Py::new(py, $wrapper { history })?.into())
        }
    };
}

set_linkage_wrapper!(set_agnes_f32, set_agnes, f32, MergeHistoryF32);
set_linkage_wrapper!(set_agnes_f64, set_agnes, f64, MergeHistoryF64);
set_linkage_wrapper!(set_anderberg_f32, set_anderberg, f32, MergeHistoryF32);
set_linkage_wrapper!(set_anderberg_f64, set_anderberg, f64, MergeHistoryF64);
set_linkage_wrapper!(set_muellner_f32, set_muellner, f32, MergeHistoryF32);
set_linkage_wrapper!(set_muellner_f64, set_muellner, f64, MergeHistoryF64);
set_linkage_wrapper!(set_nn_chain_f32, set_nn_chain, f32, MergeHistoryF32);
set_linkage_wrapper!(set_nn_chain_f64, set_nn_chain, f64, MergeHistoryF64);

// hausdorff and medoid are just wrappers around set_linkage anyway

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(agnes_f32))?;
    m.add_wrapped(wrap_pyfunction!(agnes_f64))?;
    m.add_wrapped(wrap_pyfunction!(anderberg_f32))?;
    m.add_wrapped(wrap_pyfunction!(anderberg_f64))?;
    m.add_wrapped(wrap_pyfunction!(muellner_f32))?;
    m.add_wrapped(wrap_pyfunction!(muellner_f64))?;
    m.add_wrapped(wrap_pyfunction!(nn_chain_f32))?;
    m.add_wrapped(wrap_pyfunction!(nn_chain_f64))?;
    m.add_wrapped(wrap_pyfunction!(geometric_nn_chain_f32))?;
    m.add_wrapped(wrap_pyfunction!(geometric_nn_chain_f64))?;
    m.add_wrapped(wrap_pyfunction!(incremental_nn_chain_f32))?;
    m.add_wrapped(wrap_pyfunction!(incremental_nn_chain_f64))?;
    m.add_wrapped(wrap_pyfunction!(set_agnes_f32))?;
    m.add_wrapped(wrap_pyfunction!(set_agnes_f64))?;
    m.add_wrapped(wrap_pyfunction!(set_anderberg_f32))?;
    m.add_wrapped(wrap_pyfunction!(set_anderberg_f64))?;
    m.add_wrapped(wrap_pyfunction!(set_muellner_f32))?;
    m.add_wrapped(wrap_pyfunction!(set_muellner_f64))?;
    m.add_wrapped(wrap_pyfunction!(set_nn_chain_f32))?;
    m.add_wrapped(wrap_pyfunction!(set_nn_chain_f64))?;
    m.add_wrapped(wrap_pyfunction!(heap_of_searchers_single_link_f32))?;
    m.add_wrapped(wrap_pyfunction!(heap_of_searchers_single_link_f64))?;
    m.add_wrapped(wrap_pyfunction!(lazy_buffered_search_single_link_f32))?;
    m.add_wrapped(wrap_pyfunction!(lazy_buffered_search_single_link_f64))?;
    m.add_wrapped(wrap_pyfunction!(restarting_search_single_link_f32))?;
    m.add_wrapped(wrap_pyfunction!(restarting_search_single_link_f64))?;
    m.add_wrapped(wrap_pyfunction!(buffered_search_single_link_f32))?;
    m.add_wrapped(wrap_pyfunction!(buffered_search_single_link_f64))?;
    m.add_wrapped(wrap_pyfunction!(boruvka_searchers_single_link_f32))?;
    m.add_wrapped(wrap_pyfunction!(boruvka_searchers_single_link_f64))?;
    m.add_wrapped(wrap_pyfunction!(slink_f32))?;
    m.add_wrapped(wrap_pyfunction!(slink_f64))?;
    m.add_wrapped(wrap_pyfunction!(clink_f32))?;
    m.add_wrapped(wrap_pyfunction!(clink_f64))?;
    m.add_class::<MergeHistoryF32>()?;
    m.add_class::<MergeHistoryF64>()?;
    Ok(())
}
