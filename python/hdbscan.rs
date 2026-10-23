use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::cluster::coresg::CoreSG;
use crate::cluster::hdbscan;
use crate::cluster::hdbscan::extraction::{
    extract_clusters_with_noise, extract_hdbscan_hierarchy_hdbscan,
    extract_simplified_hierarchy_hdbscan,
};
use crate::distance::DistanceFunction;
use crate::python::search::SearchIndex;
use crate::{Float, NdArrayDatasetWithDistance};

// ---- result wrappers -------------------------------------------------------

#[pyclass]
struct HdbscanHierarchyF32 {
    inner: hdbscan::HdbscanHierarchy<f32>,
}

#[pyclass]
struct HdbscanHierarchyF64 {
    inner: hdbscan::HdbscanHierarchy<f64>,
}

#[pyclass]
struct CoreSGF32 {
    inner: CoreSG<f32>,
    distance: String,
}

#[pyclass]
struct CoreSGF64 {
    inner: CoreSG<f64>,
    distance: String,
}

#[pyclass]
struct CoreSGEdgeIterF32 {
    graph: Py<CoreSGF32>,
    position: usize,
    end: usize,
}

#[pyclass]
struct CoreSGEdgeIterF64 {
    graph: Py<CoreSGF64>,
    position: usize,
    end: usize,
}

macro_rules! hdbscan_hierarchy_methods {
    ($name:ident, $float:ty) => {
        #[pymethods]
        impl $name {
            fn core_distances<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let v: Vec<$float> = self.inner.core_distances.clone();
                PyArray1::from_vec(py, v).into_py_any(py)
            }

            fn to_scipy_linkage<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
                let n = self.inner.merges.len();
                let array = PyArray2::<$float>::zeros(py, [n, 4], false);
                let mut view = unsafe { array.as_array_mut() };
                for i in 0..n {
                    view[(i, 0)] = self.inner.merges.idx1[i] as $float;
                    view[(i, 1)] = self.inner.merges.idx2[i] as $float;
                    view[(i, 2)] = self.inner.merges.distance[i];
                    view[(i, 3)] = self.inner.merges.size[i] as $float;
                }
                Ok(array.into_pyobject(py)?.into())
            }

            fn extract_clusters_with_noise<'py>(
                &self, py: Python<'py>, num_clusters: usize, min_cluster_size: usize,
            ) -> PyResult<Py<PyAny>> {
                let labels =
                    extract_clusters_with_noise(&self.inner.merges, num_clusters, min_cluster_size);
                let v: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
                PyArray1::from_vec(py, v).into_py_any(py)
            }

            fn extract_simplified<'py>(
                &self, py: Python<'py>, min_cluster_size: usize,
            ) -> PyResult<Py<PyAny>> {
                use pyo3::types::{PyDict, PyList};
                let h = extract_simplified_hierarchy_hdbscan(&self.inner, min_cluster_size);
                let nodes_list = PyList::empty(py);
                for node in h.nodes {
                    let d = PyDict::new(py);
                    d.set_item("distance", node.distance as $float)?;
                    let members: Vec<i64> = node.members.iter().map(|&v| v as i64).collect();
                    d.set_item(
                        "members",
                        PyArray1::from_vec(py, members).into_pyobject(py)?.into_any(),
                    )?;
                    let children: Vec<i64> = node.children.iter().map(|&v| v as i64).collect();
                    d.set_item(
                        "children",
                        PyArray1::from_vec(py, children).into_pyobject(py)?.into_any(),
                    )?;
                    nodes_list.append(d)?;
                }
                let roots: Vec<i64> = h.roots.iter().map(|&v| v as i64).collect();
                let result = PyDict::new(py);
                result.set_item("nodes", nodes_list)?;
                result.set_item(
                    "roots",
                    PyArray1::from_vec(py, roots).into_pyobject(py)?.into_any(),
                )?;
                result.into_py_any(py)
            }

            fn extract_hdbscan<'py>(
                &self, py: Python<'py>, min_cluster_size: usize, hierarchical: bool,
            ) -> PyResult<Py<PyAny>> {
                use pyo3::types::{PyDict, PyList};
                let r =
                    extract_hdbscan_hierarchy_hdbscan(&self.inner, min_cluster_size, hierarchical);
                let nodes_list = PyList::empty(py);
                for node in r.hierarchy.nodes {
                    let d = PyDict::new(py);
                    d.set_item("distance", node.distance as $float)?;
                    let members: Vec<i64> = node.members.iter().map(|&v| v as i64).collect();
                    d.set_item(
                        "members",
                        PyArray1::from_vec(py, members).into_pyobject(py)?.into_any(),
                    )?;
                    let children: Vec<i64> = node.children.iter().map(|&v| v as i64).collect();
                    d.set_item(
                        "children",
                        PyArray1::from_vec(py, children).into_pyobject(py)?.into_any(),
                    )?;
                    nodes_list.append(d)?;
                }
                let roots: Vec<i64> = r.hierarchy.roots.iter().map(|&v| v as i64).collect();
                let hier = PyDict::new(py);
                hier.set_item("nodes", nodes_list)?;
                hier.set_item(
                    "roots",
                    PyArray1::from_vec(py, roots).into_pyobject(py)?.into_any(),
                )?;
                let glosh: Vec<$float> = r.glosh.into_iter().map(|v| v as $float).collect();
                let result = PyDict::new(py);
                result.set_item("hierarchy", hier)?;
                result.set_item(
                    "glosh",
                    PyArray1::from_vec(py, glosh).into_pyobject(py)?.into_any(),
                )?;
                result.into_py_any(py)
            }

            fn glosh<'py>(&self, py: Python<'py>, min_cluster_size: usize) -> PyResult<Py<PyAny>> {
                use pyo3::types::PyDict;
                let r = crate::outlier::glosh::glosh(&self.inner, min_cluster_size);
                let scores: Vec<$float> = r.scores.into_iter().map(|v| v as $float).collect();
                let scores = PyArray1::from_vec(py, scores);
                let m = r.metadata;
                let meta = PyDict::new(py);
                meta.set_item("label", m.label)?;
                meta.set_item("ascending", m.ascending)?;
                meta.set_item("baseline", m.baseline as f64)?;
                meta.set_item("minimum", m.minimum as f64)?;
                meta.set_item("maximum", m.maximum as f64)?;
                meta.set_item("theoretical_minimum", m.theoretical_minimum as f64)?;
                meta.set_item("theoretical_maximum", m.theoretical_maximum as f64)?;
                (scores, meta).into_py_any(py)
            }
        }
    };
}

hdbscan_hierarchy_methods!(HdbscanHierarchyF32, f32);
hdbscan_hierarchy_methods!(HdbscanHierarchyF64, f64);

// ---- dataset / tree helpers ------------------------------------------------

fn build_dataset<'a, N>(
    array: &'a ArrayView2<'a, N>, distance: &str,
) -> PyResult<
    NdArrayDatasetWithDistance<'a, N, ArrayView2<'a, N>, Box<dyn DistanceFunction<[N], N> + Sync>>,
>
where
    N: Float,
{
    let dist_fn: Box<dyn DistanceFunction<[N], N> + Sync> = super::parse_distance_fn(distance)?;
    Ok(NdArrayDatasetWithDistance::with_distance(array, dist_fn))
}

macro_rules! coresg_python_api {
    ($graph:ident, $iterator:ident, $dtype:ty, $hierarchy:ident, $name:ident) => {
        #[pymethods]
        impl $graph {
            #[getter]
            fn max_points(&self) -> usize { self.inner.max_points }

            #[getter]
            fn num_edges(&self) -> usize { self.inner.num_edges() }

            #[getter]
            fn num_points(&self) -> usize { self.inner.core_distances.len() }

            fn __len__(&self) -> usize { self.inner.num_edges() }

            /// Iterate over (target, weight) adjacency entries for one source.
            fn neighbors(slf: Py<Self>, py: Python<'_>, source: usize) -> PyResult<Py<$iterator>> {
                let (position, end) = {
                    let graph = slf.borrow(py);
                    let neighbors = graph.inner.neighbors(source).ok_or_else(|| {
                        pyo3::exceptions::PyIndexError::new_err("source node out of range")
                    })?;
                    let position = graph.inner.offsets[source];
                    (position, position + neighbors.len())
                };
                Py::new(py, $iterator { graph: slf, position, end })
            }

            /// Return the core distance for a single point.
            fn core_distance(&self, index: usize) -> PyResult<$dtype> {
                self.inner.core_distances.get(index).copied().ok_or_else(|| {
                    pyo3::exceptions::PyIndexError::new_err("point index out of range")
                })
            }

            /// Compute a hierarchy for min_points using a caller-supplied index.
            #[pyo3(signature = (data, min_points, *, index))]
            fn hdbscan<'py>(
                &self, py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, min_points: usize,
                index: PyRef<'_, SearchIndex>,
            ) -> PyResult<Py<PyAny>> {
                if min_points == 0 || min_points > self.inner.max_points {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "min_points must be in 1..={}, got {}",
                        self.inner.max_points, min_points
                    )));
                }
                let array = data.as_array();
                if array.nrows() != self.inner.core_distances.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "data row count must match the CORE-SG graph",
                    ));
                }
                let dataset = build_dataset::<$dtype>(&array, &self.distance)?;
                let search_index = index.inner();
                let graph = &self.inner;
                let hierarchy = py
                    .detach(|| {
                        crate::reset_interrupted();
                        graph.hdbscan_hierarchy_for_min_points(search_index, &dataset, min_points)
                    })
                    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
                Py::new(py, $hierarchy { inner: hierarchy })?.into_py_any(py)
            }
        }

        #[pymethods]
        impl $iterator {
            fn __iter__(slf: Py<Self>) -> Py<Self> { slf }

            fn __next__(&mut self, py: Python<'_>) -> Option<(usize, $dtype)> {
                if self.position >= self.end {
                    return None;
                }
                let graph = self.graph.borrow(py);
                let edge = graph.inner.adjacency.get(self.position)?;
                self.position += 1;
                Some((edge.target as usize, edge.weight))
            }
        }

        #[pyfunction]
        #[pyo3(signature = (data, max_points, *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, max_points: usize,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            if max_points == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "max_points must be greater than 0",
                ));
            }
            let array = data.as_array();
            let dataset = build_dataset::<$dtype>(&array, distance)?;
            let search_index = index.inner();
            let graph = py
                .detach(|| {
                    crate::reset_interrupted();
                    crate::cluster::coresg::coresg(search_index, &dataset, max_points)
                })
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
            Py::new(py, $graph { inner: graph, distance: distance.to_owned() })?.into_py_any(py)
        }
    };
}

coresg_python_api!(CoreSGF32, CoreSGEdgeIterF32, f32, HdbscanHierarchyF32, coresg_graph_f32);
coresg_python_api!(CoreSGF64, CoreSGEdgeIterF64, f64, HdbscanHierarchyF64, coresg_graph_f64);

// ---- brute-force algorithms (no tree) --------------------------------------

macro_rules! hdbscan_brute_force_wrapper {
    ($name:ident, $algo:path, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, min_points, distance))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, min_points: usize, distance: &str,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_dataset::<$dtype>(&array, distance)?;
            let inner = py
                .detach(|| {
                    crate::reset_interrupted();
                    $algo(&dataset, min_points)
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Py::new(py, $wrapper { inner })?.into_py_any(py)
        }
    };
}

hdbscan_brute_force_wrapper!(hdbscan_prim_f32, hdbscan::hdbscan_prim, f32, HdbscanHierarchyF32);
hdbscan_brute_force_wrapper!(hdbscan_prim_f64, hdbscan::hdbscan_prim, f64, HdbscanHierarchyF64);
hdbscan_brute_force_wrapper!(slink_hdbscan_f32, hdbscan::slink_hdbscan, f32, HdbscanHierarchyF32);
hdbscan_brute_force_wrapper!(slink_hdbscan_f64, hdbscan::slink_hdbscan, f64, HdbscanHierarchyF64);

// ---- tree-accelerated algorithms (no slack) --------------------------------

macro_rules! hdbscan_tree_wrapper {
    ($name:ident, $algo:path, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, min_points, *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, min_points: usize,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_dataset::<$dtype>(&array, distance)?;
            let index = index.inner();
            let inner = py
                .detach(|| {
                    crate::reset_interrupted();
                    $algo(index, &dataset, min_points)
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Py::new(py, $wrapper { inner })?.into_py_any(py)
        }
    };
}

hdbscan_tree_wrapper!(
    heap_of_searchers_hdbscan_f32,
    hdbscan::heap_of_searchers_hdbscan,
    f32,
    HdbscanHierarchyF32
);
hdbscan_tree_wrapper!(
    heap_of_searchers_hdbscan_f64,
    hdbscan::heap_of_searchers_hdbscan,
    f64,
    HdbscanHierarchyF64
);
hdbscan_tree_wrapper!(
    restarting_search_hdbscan_f32,
    hdbscan::restarting_search_hdbscan,
    f32,
    HdbscanHierarchyF32
);
hdbscan_tree_wrapper!(
    restarting_search_hdbscan_f64,
    hdbscan::restarting_search_hdbscan,
    f64,
    HdbscanHierarchyF64
);
hdbscan_tree_wrapper!(
    boruvka_searchers_hdbscan_f32,
    hdbscan::boruvka_searchers_hdbscan,
    f32,
    HdbscanHierarchyF32
);
hdbscan_tree_wrapper!(
    boruvka_searchers_hdbscan_f64,
    hdbscan::boruvka_searchers_hdbscan,
    f64,
    HdbscanHierarchyF64
);

// ---- tree-accelerated algorithms (with slack) ------------------------------

macro_rules! hdbscan_tree_slack_wrapper {
    ($name:ident, $algo:path, $dtype:ty, $wrapper:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, min_points, slack, *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, min_points: usize, slack: usize,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_dataset::<$dtype>(&array, distance)?;
            let index = index.inner();
            let inner = py
                .detach(|| {
                    crate::reset_interrupted();
                    $algo(index, &dataset, min_points, slack)
                })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
            Py::new(py, $wrapper { inner })?.into_py_any(py)
        }
    };
}

hdbscan_tree_slack_wrapper!(
    buffered_search_hdbscan_f32,
    hdbscan::buffered_search_hdbscan,
    f32,
    HdbscanHierarchyF32
);
hdbscan_tree_slack_wrapper!(
    buffered_search_hdbscan_f64,
    hdbscan::buffered_search_hdbscan,
    f64,
    HdbscanHierarchyF64
);
hdbscan_tree_slack_wrapper!(
    lazy_buffered_search_hdbscan_f32,
    hdbscan::lazy_buffered_search_hdbscan,
    f32,
    HdbscanHierarchyF32
);
hdbscan_tree_slack_wrapper!(
    lazy_buffered_search_hdbscan_f64,
    hdbscan::lazy_buffered_search_hdbscan,
    f64,
    HdbscanHierarchyF64
);

// ---- module registration ---------------------------------------------------

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(coresg_graph_f32))?;
    m.add_wrapped(wrap_pyfunction!(coresg_graph_f64))?;
    m.add_wrapped(wrap_pyfunction!(hdbscan_prim_f32))?;
    m.add_wrapped(wrap_pyfunction!(hdbscan_prim_f64))?;
    m.add_wrapped(wrap_pyfunction!(slink_hdbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(slink_hdbscan_f64))?;
    m.add_wrapped(wrap_pyfunction!(heap_of_searchers_hdbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(heap_of_searchers_hdbscan_f64))?;
    m.add_wrapped(wrap_pyfunction!(restarting_search_hdbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(restarting_search_hdbscan_f64))?;
    m.add_wrapped(wrap_pyfunction!(boruvka_searchers_hdbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(boruvka_searchers_hdbscan_f64))?;
    m.add_wrapped(wrap_pyfunction!(buffered_search_hdbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(buffered_search_hdbscan_f64))?;
    m.add_wrapped(wrap_pyfunction!(lazy_buffered_search_hdbscan_f32))?;
    m.add_wrapped(wrap_pyfunction!(lazy_buffered_search_hdbscan_f64))?;
    m.add_class::<HdbscanHierarchyF32>()?;
    m.add_class::<HdbscanHierarchyF64>()?;
    m.add_class::<CoreSGF32>()?;
    m.add_class::<CoreSGF64>()?;
    m.add_class::<CoreSGEdgeIterF32>()?;
    m.add_class::<CoreSGEdgeIterF64>()?;
    Ok(())
}
