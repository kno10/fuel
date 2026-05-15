use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};

use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rand::SeedableRng;
use rand::distr::{Distribution, StandardUniform};
use rand_pcg::Pcg32;

use crate::cluster::kmeans::init::{FirstK, Initialization, KGeometricPP, KMeansPP, RandomSample};
use crate::cluster::kmeans::util::Centers;
use crate::{Float, NdArrayDataset, cluster};

pub(super) fn result_to_py_f32<'py>(
    py: Python<'py>, res: cluster::kmeans::util::KMeansResult<f32>,
) -> PyResult<Py<PyAny>> {
    let centers = PyArray2::from_owned_array(py, res.centers);
    let assignments = res.assignments.into_iter().map(|v| v as i64).collect::<Vec<_>>();
    let assignments = PyArray1::from_vec(py, assignments);
    let result =
        (centers, assignments, res.iterations, res.inertia, res.inertia_bound).into_pyobject(py)?;
    Ok(result.into())
}

pub(super) fn result_to_py_f64<'py>(
    py: Python<'py>, res: cluster::kmeans::util::KMeansResult<f64>,
) -> PyResult<Py<PyAny>> {
    let centers = PyArray2::from_owned_array(py, res.centers);
    let assignments = res.assignments.into_iter().map(|v| v as i64).collect::<Vec<_>>();
    let assignments = PyArray1::from_vec(py, assignments);
    let result =
        (centers, assignments, res.iterations, res.inertia, res.inertia_bound).into_pyobject(py)?;
    Ok(result.into())
}

/// Shared parameters for all k-means variants.
///
/// Not all variants use every field: `steps`, if exposed, is algorithm-specific and passed
/// separately. The common fields are k, max_iter, tol, seed, and init.
#[pyclass(get_all, set_all)]
pub struct KMeansParams {
    /// Number of clusters.
    pub k: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance (relative change in inertia or center shift).
    /// Stored as f64; cast to the algorithm's float type at call time.
    pub tol: f64,
    /// Optional RNG seed for reproducibility.
    pub seed: Option<u64>,
    /// Initialization method: 'random', 'first', 'kmeans++', 'kgeometric++',
    /// or a 2-D numpy array of shape (k, d) for fixed initial centers.
    pub init: Option<Py<PyAny>>,
}

#[pymethods]
impl KMeansParams {
    #[new]
    #[pyo3(signature = (k, max_iter=300, tol=1e-4, seed=None, init=None))]
    pub fn new(
        k: usize, max_iter: usize, tol: f64, seed: Option<u64>, init: Option<Py<PyAny>>,
    ) -> Self {
        Self { k, max_iter, tol, seed, init }
    }
}

pub(super) enum InitMethod<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
{
    Random(RandomSample<N, Pcg32>),
    First(FirstK<N>),
    KMeansPP(KMeansPP<N, Pcg32>),
    KGeometricPP(KGeometricPP<N, Pcg32>),
    Fixed(Centers<N>),
}

impl<N> InitMethod<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum + Element,
    StandardUniform: Distribution<N>,
{
    fn from_str(init: &str, seed: u64) -> PyResult<Self> {
        match init {
            "random" => Ok(InitMethod::Random(RandomSample::new(Pcg32::seed_from_u64(seed)))),
            "first" => Ok(InitMethod::First(FirstK::new())),
            "kmeans++" => Ok(InitMethod::KMeansPP(KMeansPP::new(Pcg32::seed_from_u64(seed)))),
            "kgeometric++" => {
                Ok(InitMethod::KGeometricPP(KGeometricPP::new(Pcg32::seed_from_u64(seed))))
            }
            other => Err(PyValueError::new_err(format!(
                "unknown initialization method '{}', valid options are 'random', 'first', 'kmeans++', 'kgeometric++'",
                other
            ))),
        }
    }

    pub(super) fn parse<'py>(
        py: Python<'py>, init: Option<Py<PyAny>>, k: usize, d: usize, seed: Option<u64>,
    ) -> PyResult<Self> {
        let seed = seed.unwrap_or(0);
        if let Some(init) = init {
            let init = init.as_ref().as_any();
            if let Ok(init_str) = init.extract::<&str>(py) {
                return InitMethod::from_str(init_str, seed);
            }

            let arr = init
                .extract::<PyReadonlyArray2<N>>(py)
                .map_err(|_| PyValueError::new_err("init must be a string or a 2D numpy array"))?;
            let arr = arr.as_array();
            let (n0, n1) = arr.dim();
            if n0 != k || n1 != d {
                return Err(PyValueError::new_err(format!(
                    "init array must have shape (k, d) = ({}, {}), got ({}, {})",
                    k, d, n0, n1
                )));
            }
            let slice = arr.as_slice().ok_or_else(|| {
                PyValueError::new_err("init array must be contiguous and owned by numpy")
            })?;
            let mut centers = Centers::new(k, d);
            for i in 0..k {
                centers.center_mut(i).copy_from_slice(&slice[i * d..(i + 1) * d]);
            }
            return Ok(InitMethod::Fixed(centers));
        }

        InitMethod::from_str("random", seed)
    }
}

impl<N> Initialization<N> for InitMethod<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign + Sum,
    StandardUniform: Distribution<N>,
{
    fn uses_distances(&self) -> bool {
        match self {
            InitMethod::Random(r) => r.uses_distances(),
            InitMethod::First(f) => f.uses_distances(),
            InitMethod::KMeansPP(km) => km.uses_distances(),
            InitMethod::KGeometricPP(km) => km.uses_distances(),
            InitMethod::Fixed(_) => false,
        }
    }

    fn init<A>(&mut self, data: &A, cent: &mut Centers<N>, k: usize)
    where
        A: crate::VectorData<N>,
    {
        match self {
            InitMethod::Random(r) => r.init(data, cent, k),
            InitMethod::First(f) => f.init(data, cent, k),
            InitMethod::KMeansPP(km) => km.init(data, cent, k),
            InitMethod::KGeometricPP(km) => km.init(data, cent, k),
            InitMethod::Fixed(fixed) => {
                for i in 0..k {
                    cent.center_mut(i).copy_from_slice(fixed.center(i));
                }
            }
        }
    }

    fn init_with_distances<A, F>(
        &mut self, data: &A, cent: &mut Centers<N>, k: usize, callback: Option<F>,
    ) where
        A: crate::VectorData<N>,
        F: FnMut(usize, usize, N),
    {
        match self {
            InitMethod::Random(r) => r.init_with_distances(data, cent, k, callback),
            InitMethod::First(f) => f.init_with_distances(data, cent, k, callback),
            InitMethod::KMeansPP(km) => km.init_with_distances(data, cent, k, callback),
            InitMethod::KGeometricPP(km) => km.init_with_distances(data, cent, k, callback),
            InitMethod::Fixed(fixed) => {
                for i in 0..k {
                    cent.center_mut(i).copy_from_slice(fixed.center(i));
                }
            }
        }
    }
}

macro_rules! variant_call {
    ($name:ident, $variant:ident, $dtype:ty, $result_fn:ident) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, params: &KMeansParams,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let k = params.k;
            let dataset = NdArrayDataset::new(&array);
            let mut init = InitMethod::parse(
                py,
                params.init.as_ref().map(|x| x.clone_ref(py)),
                k,
                array.ncols(),
                params.seed,
            )?;
            let tol = <$dtype as Float>::cast(params.tol);
            let res = crate::py_interruptible(py, move || {
                crate::cluster::kmeans::$variant::<$dtype, _, _>(
                    &dataset,
                    k,
                    &mut init,
                    params.max_iter,
                    tol,
                )
                .map_err(|e| format!("{e:?}"))
            })?;
            $result_fn(py, res)
        }
    };
}

macro_rules! variant_call_steps {
    ($name:ident, $variant:ident, $dtype:ty, $result_fn:ident) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, params: &KMeansParams,
            steps: usize,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let k = params.k;
            let dataset = NdArrayDataset::new(&array);
            let mut init = InitMethod::parse(
                py,
                params.init.as_ref().map(|x| x.clone_ref(py)),
                k,
                array.ncols(),
                params.seed,
            )?;
            let tol = <$dtype as Float>::cast(params.tol);
            let res = crate::py_interruptible(py, move || {
                crate::cluster::kmeans::$variant::<$dtype, _, _>(
                    &dataset,
                    k,
                    &mut init,
                    params.max_iter,
                    tol,
                    steps,
                )
                .map_err(|e| format!("{e:?}"))
            })?;
            $result_fn(py, res)
        }
    };
}

macro_rules! variant_call_p {
    ($name:ident, $variant:ident, $dtype:ty, $result_fn:ident) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, params: &KMeansParams, p: $dtype,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let k = params.k;
            let dataset = NdArrayDataset::new(&array);
            let mut init = InitMethod::parse(
                py,
                params.init.as_ref().map(|x| x.clone_ref(py)),
                k,
                array.ncols(),
                params.seed,
            )?;
            let tol = <$dtype as Float>::cast(params.tol);
            let res = crate::py_interruptible(py, move || {
                crate::cluster::kmeans::$variant::<$dtype, _, _>(
                    &dataset,
                    k,
                    &mut init,
                    params.max_iter,
                    tol,
                    p,
                )
                .map_err(|e| format!("{e:?}"))
            })?;
            $result_fn(py, res)
        }
    };
}

macro_rules! variant_call_pp {
    ($name:ident, $variant:ident, $dtype:ty, $result_fn:ident) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, params: &KMeansParams,
            gamma: $dtype, alpha: $dtype,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let k = params.k;
            let dataset = NdArrayDataset::new(&array);
            let mut init = InitMethod::parse(
                py,
                params.init.as_ref().map(|x| x.clone_ref(py)),
                k,
                array.ncols(),
                params.seed,
            )?;
            let tol = <$dtype as Float>::cast(params.tol);
            let res = crate::py_interruptible(py, move || {
                crate::cluster::kmeans::$variant::<$dtype, _, _>(
                    &dataset,
                    k,
                    &mut init,
                    params.max_iter,
                    tol,
                    gamma,
                    alpha,
                )
                .map_err(|e| format!("{e:?}"))
            })?;
            $result_fn(py, res)
        }
    };
}

macro_rules! variant_call_alpha {
    ($name:ident, $variant:ident, $dtype:ty, $result_fn:ident) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, params: &KMeansParams, alpha: f32,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let k = params.k;
            let dataset = NdArrayDataset::new(&array);
            let mut init = InitMethod::parse(
                py,
                params.init.as_ref().map(|x| x.clone_ref(py)),
                k,
                array.ncols(),
                params.seed,
            )?;
            let tol = <$dtype as Float>::cast(params.tol);
            let res = crate::py_interruptible(py, move || {
                crate::cluster::kmeans::$variant::<$dtype, _, _>(
                    &dataset,
                    k,
                    &mut init,
                    params.max_iter,
                    tol,
                    alpha,
                )
                .map_err(|e| format!("{e:?}"))
            })?;
            $result_fn(py, res)
        }
    };
}

variant_call!(lloyd_f32, lloyd, f32, result_to_py_f32);
variant_call!(lloyd_f64, lloyd, f64, result_to_py_f64);
variant_call!(elkan_f32, elkan, f32, result_to_py_f32);
variant_call!(elkan_f64, elkan, f64, result_to_py_f64);
variant_call!(hamerly_f32, hamerly, f32, result_to_py_f32);
variant_call!(hamerly_f64, hamerly, f64, result_to_py_f64);
variant_call!(hartigan_wong_f32, hartigan_wong, f32, result_to_py_f32);
variant_call!(hartigan_wong_f64, hartigan_wong, f64, result_to_py_f64);
variant_call!(macqueen_f32, macqueen, f32, result_to_py_f32);
variant_call!(macqueen_f64, macqueen, f64, result_to_py_f64);
variant_call!(shallot_f32, shallot, f32, result_to_py_f32);
variant_call!(shallot_f64, shallot, f64, result_to_py_f64);
variant_call!(exponion_f32, exponion, f32, result_to_py_f32);
variant_call!(exponion_f64, exponion, f64, result_to_py_f64);
variant_call!(kmedians_f32, kmedians, f32, result_to_py_f32);
variant_call!(kmedians_f64, kmedians, f64, result_to_py_f64);
variant_call_pp!(kgmedians_f32, kgmedians, f32, result_to_py_f32);
variant_call_pp!(kgmedians_f64, kgmedians, f64, result_to_py_f64);
variant_call_steps!(kgeometric_f32, kgeometric, f32, result_to_py_f32);
variant_call_steps!(kgeometric_f64, kgeometric, f64, result_to_py_f64);
variant_call_steps!(kgeometric_sh_f32, kgeometric_sh, f32, result_to_py_f32);
variant_call_steps!(kgeometric_sh_f64, kgeometric_sh, f64, result_to_py_f64);
variant_call_p!(kharmonic_f32, kharmonic, f32, result_to_py_f32);
variant_call_p!(kharmonic_f64, kharmonic, f64, result_to_py_f64);
variant_call_alpha!(tkmeans_f32, tkmeans, f32, result_to_py_f32);
variant_call_alpha!(tkmeans_f64, tkmeans, f64, result_to_py_f64);
variant_call!(lloyd_blas_f32, lloyd_blas, f32, result_to_py_f32);
variant_call!(lloyd_blas_f64, lloyd_blas, f64, result_to_py_f64);
variant_call!(lloyd_naive_f32, lloyd_naive, f32, result_to_py_f32);
variant_call!(lloyd_naive_f64, lloyd_naive, f64, result_to_py_f64);
variant_call!(hartigan_wong_quick_f32, hartigan_wong_quick, f32, result_to_py_f32);
variant_call!(hartigan_wong_quick_f64, hartigan_wong_quick, f64, result_to_py_f64);
variant_call!(simp_elkan_f32, simp_elkan, f32, result_to_py_f32);
variant_call!(simp_elkan_f64, simp_elkan, f64, result_to_py_f64);
variant_call!(simp_hamerly_f32, simp_hamerly, f32, result_to_py_f32);
variant_call!(simp_hamerly_f64, simp_hamerly, f64, result_to_py_f64);

fn result_to_py_fuzzy<'py, N>(
    py: Python<'py>, centers: ndarray::Array2<N>, membership: ndarray::Array2<N>,
    assignments: Vec<usize>, iterations: usize, loss: N,
) -> PyResult<Py<PyAny>>
where
    N: numpy::Element + Copy + pyo3::IntoPyObject<'py>,
{
    let centers = PyArray2::from_owned_array(py, centers);
    let membership = PyArray2::from_owned_array(py, membership);
    let assignments = assignments.into_iter().map(|v| v as i64).collect::<Vec<_>>();
    let assignments = PyArray1::from_vec(py, assignments);
    Ok((centers, membership, assignments, iterations, loss).into_pyobject(py)?.into())
}

#[pyfunction]
fn fuzzy_lloyd_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, params: &KMeansParams, m: f32,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let k = params.k;
    let dataset = NdArrayDataset::new(&array);
    let mut init = InitMethod::parse(
        py,
        params.init.as_ref().map(|x| x.clone_ref(py)),
        k,
        array.ncols(),
        params.seed,
    )?;
    let (centers, membership, assignments, iterations, loss) = crate::py_interruptible(py, move || {
        Ok(cluster::kmeans::fuzzy_lloyd::<f32, _, _>(&dataset, k, &mut init, params.max_iter, m))
    })?;
    result_to_py_fuzzy(py, centers, membership, assignments, iterations, loss)
}

#[pyfunction]
fn fuzzy_lloyd_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, params: &KMeansParams, m: f64,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let k = params.k;
    let dataset = NdArrayDataset::new(&array);
    let mut init = InitMethod::parse(
        py,
        params.init.as_ref().map(|x| x.clone_ref(py)),
        k,
        array.ncols(),
        params.seed,
    )?;
    let (centers, membership, assignments, iterations, loss) = crate::py_interruptible(py, move || {
        Ok(cluster::kmeans::fuzzy_lloyd::<f64, _, _>(&dataset, k, &mut init, params.max_iter, m))
    })?;
    result_to_py_fuzzy(py, centers, membership, assignments, iterations, loss)
}

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<KMeansParams>()?;
    m.add_wrapped(wrap_pyfunction!(lloyd_f32))?;
    m.add_wrapped(wrap_pyfunction!(lloyd_f64))?;
    m.add_wrapped(wrap_pyfunction!(elkan_f32))?;
    m.add_wrapped(wrap_pyfunction!(elkan_f64))?;
    m.add_wrapped(wrap_pyfunction!(hamerly_f32))?;
    m.add_wrapped(wrap_pyfunction!(hamerly_f64))?;
    m.add_wrapped(wrap_pyfunction!(hartigan_wong_f32))?;
    m.add_wrapped(wrap_pyfunction!(hartigan_wong_f64))?;
    m.add_wrapped(wrap_pyfunction!(macqueen_f32))?;
    m.add_wrapped(wrap_pyfunction!(macqueen_f64))?;
    m.add_wrapped(wrap_pyfunction!(shallot_f32))?;
    m.add_wrapped(wrap_pyfunction!(shallot_f64))?;
    m.add_wrapped(wrap_pyfunction!(exponion_f32))?;
    m.add_wrapped(wrap_pyfunction!(exponion_f64))?;
    m.add_wrapped(wrap_pyfunction!(kmedians_f32))?;
    m.add_wrapped(wrap_pyfunction!(kmedians_f64))?;
    m.add_wrapped(wrap_pyfunction!(kgmedians_f32))?;
    m.add_wrapped(wrap_pyfunction!(kgmedians_f64))?;
    m.add_wrapped(wrap_pyfunction!(kgeometric_f32))?;
    m.add_wrapped(wrap_pyfunction!(kgeometric_f64))?;
    m.add_wrapped(wrap_pyfunction!(kgeometric_sh_f32))?;
    m.add_wrapped(wrap_pyfunction!(kgeometric_sh_f64))?;
    m.add_wrapped(wrap_pyfunction!(kharmonic_f32))?;
    m.add_wrapped(wrap_pyfunction!(kharmonic_f64))?;
    m.add_wrapped(wrap_pyfunction!(tkmeans_f32))?;
    m.add_wrapped(wrap_pyfunction!(tkmeans_f64))?;
    m.add_wrapped(wrap_pyfunction!(lloyd_blas_f32))?;
    m.add_wrapped(wrap_pyfunction!(lloyd_blas_f64))?;
    m.add_wrapped(wrap_pyfunction!(lloyd_naive_f32))?;
    m.add_wrapped(wrap_pyfunction!(lloyd_naive_f64))?;
    m.add_wrapped(wrap_pyfunction!(hartigan_wong_quick_f32))?;
    m.add_wrapped(wrap_pyfunction!(hartigan_wong_quick_f64))?;
    m.add_wrapped(wrap_pyfunction!(simp_elkan_f32))?;
    m.add_wrapped(wrap_pyfunction!(simp_elkan_f64))?;
    m.add_wrapped(wrap_pyfunction!(simp_hamerly_f32))?;
    m.add_wrapped(wrap_pyfunction!(simp_hamerly_f64))?;
    m.add_wrapped(wrap_pyfunction!(fuzzy_lloyd_f32))?;
    m.add_wrapped(wrap_pyfunction!(fuzzy_lloyd_f64))?;
    Ok(())
}
