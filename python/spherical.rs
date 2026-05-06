use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use super::kmeans::{InitMethod, KMeansParams, result_to_py_f32, result_to_py_f64};
use super::sparse::parse_csr_dataset;
use crate::{NdArrayDataset, VectorData, cluster};

macro_rules! variant_call {
    ($name:ident, $variant:ident, $dtype:ty, $result_fn:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, params))]
        fn $name(
            py: Python<'_>, data: PyReadonlyArray2<'_, $dtype>, params: &KMeansParams,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDataset::new(&array);
            let k = params.k;
            let mut init = InitMethod::<$dtype>::parse(
                py,
                params.init.as_ref().map(|x| x.clone_ref(py)),
                k,
                array.ncols(),
                params.seed,
            )?;
            let tol = <$dtype as crate::Float>::cast(params.tol);
            let res = cluster::kmeans::$variant::<$dtype, _, _>(
                &dataset,
                k,
                &mut init,
                params.max_iter,
                tol,
            )
            .into();
            $result_fn(py, res)
        }
    };
}

macro_rules! variant_call_sparse {
    ($name:ident, $variant:ident, $dtype:ty, $result_fn:ident) => {
        #[pyfunction]
        #[pyo3(signature = (data, params))]
        fn $name(
            py: Python<'_>, data: Py<PyAny>, params: &KMeansParams,
        ) -> PyResult<Py<PyAny>> {
            let dataset = parse_csr_dataset::<$dtype>(py, data)?;
            let k = params.k;
            let mut init = InitMethod::<$dtype>::parse(
                py,
                params.init.as_ref().map(|x| x.clone_ref(py)),
                k,
                dataset.dims(),
                params.seed,
            )?;
            let tol = <$dtype as crate::Float>::cast(params.tol);
            let res = cluster::kmeans::$variant::<$dtype, _, _>(
                &dataset,
                k,
                &mut init,
                params.max_iter,
                tol,
            )
            .into();
            $result_fn(py, res)
        }
    };
}

variant_call!(spherical_lloyd_f32, spherical_lloyd, f32, result_to_py_f32);
variant_call!(spherical_lloyd_f64, spherical_lloyd, f64, result_to_py_f64);
variant_call!(spherical_elkan_f32, spherical_elkan, f32, result_to_py_f32);
variant_call!(spherical_elkan_f64, spherical_elkan, f64, result_to_py_f64);
variant_call!(spherical_simp_elkan_f32, spherical_simp_elkan, f32, result_to_py_f32);
variant_call!(spherical_simp_elkan_f64, spherical_simp_elkan, f64, result_to_py_f64);
variant_call!(spherical_hamerly_f32, spherical_hamerly, f32, result_to_py_f32);
variant_call!(spherical_hamerly_f64, spherical_hamerly, f64, result_to_py_f64);
variant_call!(spherical_simp_hamerly_f32, spherical_simp_hamerly, f32, result_to_py_f32);
variant_call!(spherical_simp_hamerly_f64, spherical_simp_hamerly, f64, result_to_py_f64);
variant_call_sparse!(spherical_lloyd_sparse_f32, spherical_lloyd, f32, result_to_py_f32);
variant_call_sparse!(spherical_lloyd_sparse_f64, spherical_lloyd, f64, result_to_py_f64);
variant_call_sparse!(spherical_elkan_sparse_f32, spherical_elkan, f32, result_to_py_f32);
variant_call_sparse!(spherical_elkan_sparse_f64, spherical_elkan, f64, result_to_py_f64);
variant_call_sparse!(spherical_simp_elkan_sparse_f32, spherical_simp_elkan, f32, result_to_py_f32);
variant_call_sparse!(spherical_simp_elkan_sparse_f64, spherical_simp_elkan, f64, result_to_py_f64);
variant_call_sparse!(spherical_hamerly_sparse_f32, spherical_hamerly, f32, result_to_py_f32);
variant_call_sparse!(spherical_hamerly_sparse_f64, spherical_hamerly, f64, result_to_py_f64);
variant_call_sparse!(
    spherical_simp_hamerly_sparse_f32,
    spherical_simp_hamerly,
    f32,
    result_to_py_f32
);
variant_call_sparse!(
    spherical_simp_hamerly_sparse_f64,
    spherical_simp_hamerly,
    f64,
    result_to_py_f64
);

pub fn register(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(spherical_lloyd_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_lloyd_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_elkan_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_elkan_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_elkan_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_elkan_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_hamerly_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_hamerly_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_hamerly_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_hamerly_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_lloyd_sparse_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_lloyd_sparse_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_elkan_sparse_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_elkan_sparse_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_elkan_sparse_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_elkan_sparse_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_hamerly_sparse_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_hamerly_sparse_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_hamerly_sparse_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_simp_hamerly_sparse_f64))?;
    Ok(())
}
