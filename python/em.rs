use std::iter::Sum;
use std::ops::{AddAssign, MulAssign, SubAssign};

use cluster::em::EmModel;
use ndarray::{Array1, Array2};
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rand_pcg::Pcg32;

use super::make_rng;
use super::sparse::parse_csr_dataset;
use crate::{Float, NdArrayDataset, cluster};

fn py_assignments<'py>(py: Python<'py>, assignments: Vec<usize>) -> PyResult<Py<PyAny>> {
    let assignments = assignments.into_iter().map(|v| v as i64).collect::<Vec<_>>();
    PyArray1::from_vec(py, assignments).into_py_any(py)
}

fn py_responsibilities<'py, N>(
    py: Python<'py>, responsibilities: Option<Array2<N>>,
) -> PyResult<Option<Py<PyAny>>>
where
    N: Element + Copy,
{
    if let Some(arr) = responsibilities {
        Ok(Some(PyArray2::from_owned_array(py, arr).into_py_any(py)?))
    } else {
        Ok(None)
    }
}

fn run_em_model<'py, N, D, Mo, F, C>(
    py: Python<'py>, dataset: &'py D, k: usize, delta: N, miniter: usize, maxiter: usize,
    hard: bool, prior: N, return_soft: bool, min_log_likelihood: N, noise_ratio: N,
    seed: Option<u64>, factory_builder: F, converter: C,
) -> PyResult<Py<PyAny>>
where
    N: Float + Element + Copy + AddAssign + SubAssign + MulAssign + Sum,
    D: crate::VectorData<N>,
    Mo: cluster::em::EmModel<N>,
    F: FnOnce(cluster::kmeans::init::RandomSample<N, Pcg32>, &D) -> Vec<Mo>,
    C: FnOnce(Python<'py>, cluster::em::EmResult<N, Mo>) -> PyResult<Py<PyAny>>,
{
    let rng = make_rng(seed);
    let init = cluster::kmeans::init::RandomSample::new(rng);
    let models = factory_builder(init, dataset);
    let config = cluster::em::EmConfig {
        delta,
        miniter,
        maxiter,
        hard,
        prior,
        return_soft,
        min_log_likelihood,
        noise_ratio,
    };
    let res = cluster::em::expectation_maximization::<N, _, _>(dataset, k, models, config);
    converter(py, res)
}

/// Diagonal-covariance Gaussian EM; variant: "default" | "textbook".
macro_rules! define_diagonal_em {
    ($name:ident, $float:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $float>, k: usize, delta: $float,
            miniter: usize, maxiter: usize, hard: bool, prior: $float, return_soft: bool,
            min_log_likelihood: $float, noise_ratio: $float, variant: &str, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDataset::new(&array);
            macro_rules! run_variant {
                ($factory_expr:expr) => {
                    run_em_model(
                        py,
                        &dataset,
                        k,
                        delta,
                        miniter,
                        maxiter,
                        hard,
                        prior,
                        return_soft,
                        min_log_likelihood,
                        noise_ratio,
                        seed,
                        $factory_expr,
                        |py, res| {
                            let mut weights = Vec::with_capacity(k);
                            let mut means = Vec::with_capacity(k);
                            let mut variances = Vec::with_capacity(k);
                            for model in res.models {
                                weights.push(model.weight());
                                means.push(
                                    PyArray1::from_owned_array(
                                        py,
                                        Array1::from_vec(model.mean().to_vec()),
                                    )
                                    .into_py_any(py)?,
                                );
                                variances.push(
                                    PyArray1::from_owned_array(
                                        py,
                                        Array1::from_vec(model.variance().to_vec()),
                                    )
                                    .into_py_any(py)?,
                                );
                            }
                            (
                                PyArray1::from_vec(py, weights).into_py_any(py)?,
                                PyList::new(py, means)?.into_py_any(py)?,
                                PyList::new(py, variances)?.into_py_any(py)?,
                                py_assignments(py, res.assignments)?,
                                py_responsibilities(py, res.responsibilities)?,
                                res.n_iter,
                                res.log_likelihood,
                            )
                                .into_py_any(py)
                        },
                    )
                };
            }
            match variant {
                "textbook" => run_variant!(|init, dataset| {
                    cluster::em::TextbookDiagonalGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
                "two_pass" => run_variant!(|init, dataset| {
                    cluster::em::TwoPassDiagonalGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
                _ => run_variant!(|init, dataset| {
                    cluster::em::DiagonalGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
            }
        }
    };
}

define_diagonal_em!(diagonal_em_f32, f32);
define_diagonal_em!(diagonal_em_f64, f64);

/// Spherical-covariance Gaussian EM; variant: "default" | "textbook".
macro_rules! define_spherical_em {
    ($name:ident, $float:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $float>, k: usize, delta: $float,
            miniter: usize, maxiter: usize, hard: bool, prior: $float, return_soft: bool,
            min_log_likelihood: $float, noise_ratio: $float, variant: &str, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDataset::new(&array);
            macro_rules! run_variant {
                ($factory_expr:expr) => {
                    run_em_model(
                        py,
                        &dataset,
                        k,
                        delta,
                        miniter,
                        maxiter,
                        hard,
                        prior,
                        return_soft,
                        min_log_likelihood,
                        noise_ratio,
                        seed,
                        $factory_expr,
                        |py, res| {
                            let mut weights = Vec::with_capacity(k);
                            let mut means = Vec::with_capacity(k);
                            let mut variances = Vec::with_capacity(k);
                            for model in res.models {
                                weights.push(model.weight());
                                means.push(
                                    PyArray1::from_owned_array(
                                        py,
                                        Array1::from_vec(model.mean().to_vec()),
                                    )
                                    .into_py_any(py)?,
                                );
                                variances.push(model.variance());
                            }
                            (
                                PyArray1::from_vec(py, weights).into_py_any(py)?,
                                PyList::new(py, means)?.into_py_any(py)?,
                                PyArray1::from_vec(py, variances).into_py_any(py)?,
                                py_assignments(py, res.assignments)?,
                                py_responsibilities(py, res.responsibilities)?,
                                res.n_iter,
                                res.log_likelihood,
                            )
                                .into_py_any(py)
                        },
                    )
                };
            }
            match variant {
                "textbook" => run_variant!(|init, dataset| {
                    cluster::em::TextbookSphericalGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
                "two_pass" => run_variant!(|init, dataset| {
                    cluster::em::TwoPassSphericalGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
                _ => run_variant!(|init, dataset| {
                    cluster::em::SphericalGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
            }
        }
    };
}

define_spherical_em!(spherical_em_f32, f32);
define_spherical_em!(spherical_em_f64, f64);

/// Full-covariance Gaussian EM; variant: "default" | "textbook" | "two_pass".
macro_rules! define_multivariate_em {
    ($name:ident, $float:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $float>, k: usize, delta: $float,
            miniter: usize, maxiter: usize, hard: bool, prior: $float, return_soft: bool,
            min_log_likelihood: $float, noise_ratio: $float, variant: &str, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDataset::new(&array);
            macro_rules! run_variant {
                ($factory_expr:expr) => {
                    run_em_model(
                        py,
                        &dataset,
                        k,
                        delta,
                        miniter,
                        maxiter,
                        hard,
                        prior,
                        return_soft,
                        min_log_likelihood,
                        noise_ratio,
                        seed,
                        $factory_expr,
                        |py, res| {
                            let mut weights = Vec::with_capacity(k);
                            let mut means = Vec::with_capacity(k);
                            let mut covariances = Vec::with_capacity(k);
                            for model in res.models {
                                let dim = model.mean().len();
                                weights.push(model.weight());
                                means.push(
                                    PyArray1::from_owned_array(
                                        py,
                                        Array1::from_vec(model.mean().to_vec()),
                                    )
                                    .into_py_any(py)?,
                                );
                                covariances.push(
                                    PyArray2::from_owned_array(
                                        py,
                                        Array2::from_shape_vec(
                                            (dim, dim),
                                            model.covariance().to_vec(),
                                        )
                                        .unwrap(),
                                    )
                                    .into_py_any(py)?,
                                );
                            }
                            (
                                PyArray1::from_vec(py, weights).into_py_any(py)?,
                                PyList::new(py, means)?.into_py_any(py)?,
                                PyList::new(py, covariances)?.into_py_any(py)?,
                                py_assignments(py, res.assignments)?,
                                py_responsibilities(py, res.responsibilities)?,
                                res.n_iter,
                                res.log_likelihood,
                            )
                                .into_py_any(py)
                        },
                    )
                };
            }
            match variant {
                "textbook" => run_variant!(|init, dataset| {
                    cluster::em::TextbookMultivariateGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
                "two_pass" => run_variant!(|init, dataset| {
                    cluster::em::TwoPassMultivariateGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
                _ => run_variant!(|init, dataset| {
                    cluster::em::MultivariateGaussianModelFactory::new(init)
                        .build_initial_models(dataset, k)
                }),
            }
        }
    };
}

define_multivariate_em!(multivariate_em_f32, f32);
define_multivariate_em!(multivariate_em_f64, f64);

/// Von Mises-Fisher EM on sparse data; variant parameter reserved (only "default").
macro_rules! define_vmf_em_sparse {
    ($name:ident, $float:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: Py<PyAny>, k: usize, delta: $float, miniter: usize,
            maxiter: usize, hard: bool, prior: $float, return_soft: bool,
            min_log_likelihood: $float, noise_ratio: $float, init_kappa: $float, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let dataset = parse_csr_dataset::<$float>(py, data)?;
            run_em_model(
                py,
                &dataset,
                k,
                delta,
                miniter,
                maxiter,
                hard,
                prior,
                return_soft,
                min_log_likelihood,
                noise_ratio,
                seed,
                |init, dataset| {
                    cluster::em::VonMisesFisherModelFactory::new(init)
                        .with_kappa(init_kappa)
                        .build_initial_models(dataset, k)
                },
                |py, res| {
                    let mut weights = Vec::with_capacity(k);
                    let mut means = Vec::with_capacity(k);
                    let mut kappas = Vec::with_capacity(k);
                    for model in res.models {
                        weights.push(model.weight());
                        means.push(
                            PyArray1::from_owned_array(py, Array1::from_vec(model.mean().to_vec()))
                                .into_py_any(py)?,
                        );
                        kappas.push(model.kappa());
                    }
                    (
                        PyArray1::from_vec(py, weights).into_py_any(py)?,
                        PyList::new(py, means)?.into_py_any(py)?,
                        PyArray1::from_vec(py, kappas).into_py_any(py)?,
                        py_assignments(py, res.assignments)?,
                        py_responsibilities(py, res.responsibilities)?,
                        res.n_iter,
                        res.log_likelihood,
                    )
                        .into_py_any(py)
                },
            )
        }
    };
}

define_vmf_em_sparse!(von_mises_fisher_em_sparse_f32, f32);
define_vmf_em_sparse!(von_mises_fisher_em_sparse_f64, f64);

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(diagonal_em_f32))?;
    m.add_wrapped(wrap_pyfunction!(diagonal_em_f64))?;
    m.add_wrapped(wrap_pyfunction!(spherical_em_f32))?;
    m.add_wrapped(wrap_pyfunction!(spherical_em_f64))?;
    m.add_wrapped(wrap_pyfunction!(multivariate_em_f32))?;
    m.add_wrapped(wrap_pyfunction!(multivariate_em_f64))?;
    m.add_wrapped(wrap_pyfunction!(von_mises_fisher_em_sparse_f32))?;
    m.add_wrapped(wrap_pyfunction!(von_mises_fisher_em_sparse_f64))?;
    Ok(())
}
