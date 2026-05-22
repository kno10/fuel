use numpy::{Element, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::distance::{DistanceFunction, Euclidean};
use crate::intrinsicdimensionality::{
    ABID, ALID, AggregatedHillID, GeneralizedExpansionDimension, HillID, LMomentsEstimator,
    MethodOfMoments, ProbabilityWeightedMoments, ProbabilityWeightedMoments2, RABID, RVEstimator,
    TightLID, ZipfID,
};
use crate::kernel::polynomial::PolynomialKernel;
use crate::outlier::cop::CopDistanceDist;
use crate::outlier::kernel::KernelDensityFunction;
use crate::python::search::SearchIndex;
use crate::{Float, NdArrayDatasetWithDistance, outlier};

fn parse_abod_kernel<F>(kernel: &str) -> PyResult<Box<dyn Fn(&[F], &[F]) -> F + Sync + Send>>
where
    F: Float + 'static,
{
    match kernel.to_lowercase().as_str() {
        "poly2" => {
            let pk = PolynomialKernel::new(2usize, F::one(), F::zero());
            Ok(Box::new(move |x, y| pk.similarity(x, y)))
        }
        "poly3" => {
            let pk = PolynomialKernel::new(3usize, F::one(), F::zero());
            Ok(Box::new(move |x, y| pk.similarity(x, y)))
        }
        "linear" => {
            let pk = PolynomialKernel::new(1usize, F::one(), F::zero());
            Ok(Box::new(move |x, y| pk.similarity(x, y)))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown ABOD kernel '{}', valid values: poly2, poly3, linear",
            other
        ))),
    }
}

fn build_outlier_dataset<'a, N>(
    array: &'a ndarray::ArrayView2<'a, N>, distance: &str,
) -> PyResult<
    NdArrayDatasetWithDistance<
        'a,
        N,
        ndarray::ArrayView2<'a, N>,
        Box<dyn DistanceFunction<[N], N> + Sync + Send>,
    >,
>
where
    N: Float,
{
    let dist_fn: Box<dyn DistanceFunction<[N], N> + Sync + Send> =
        super::parse_distance_fn(distance)?;
    Ok(NdArrayDatasetWithDistance::with_distance(array, dist_fn))
}

fn result_to_py_outlier<'py, F>(
    py: Python<'py>, result: outlier::common::OutlierResult<F>,
) -> PyResult<Py<PyAny>>
where
    F: Float + Copy + Element + pyo3::IntoPyObject<'py>,
{
    let scores = PyArray1::from_vec(py, result.scores);
    let m = result.metadata;
    let meta = pyo3::types::PyDict::new(py);
    meta.set_item("label", m.label)?;
    meta.set_item("ascending", m.ascending)?;
    meta.set_item("baseline", m.baseline)?;
    meta.set_item("minimum", m.minimum)?;
    meta.set_item("maximum", m.maximum)?;
    meta.set_item("theoretical_minimum", m.theoretical_minimum)?;
    meta.set_item("theoretical_maximum", m.theoretical_maximum)?;
    let output = (scores, meta).into_pyobject(py)?;
    Ok(output.into())
}

fn parse_cop_distance_dist(dist: &str) -> PyResult<CopDistanceDist> {
    match dist.to_lowercase().as_str() {
        "chi2" | "chi_squared" | "chisq" | "chi-squared" => Ok(CopDistanceDist::ChiSquared),
        "gamma" => Ok(CopDistanceDist::Gamma),
        other => Err(PyValueError::new_err(format!(
            "unknown COP distribution '{}', valid values are 'chi2' or 'gamma'",
            other
        ))),
    }
}

fn parse_kernel_density_function(kernel: &str) -> PyResult<KernelDensityFunction> {
    match kernel.to_lowercase().as_str() {
        "uniform" => Ok(KernelDensityFunction::Uniform),
        "triangular" => Ok(KernelDensityFunction::Triangular),
        "epanechnikov" => Ok(KernelDensityFunction::Epanechnikov),
        "biweight" => Ok(KernelDensityFunction::Biweight),
        "triweight" => Ok(KernelDensityFunction::Triweight),
        "cosine" => Ok(KernelDensityFunction::Cosine),
        "gaussian" => Ok(KernelDensityFunction::Gaussian),
        other => Err(PyValueError::new_err(format!(
            "unknown kernel '{}', valid values are: uniform, triangular, epanechnikov, biweight, triweight, cosine, gaussian",
            other
        ))),
    }
}

macro_rules! tree_outlier_p {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, perplexity, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, perplexity: f64, distance: &str,
            index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                Ok(outlier::$variant::<_, _, $dtype>(tree, &dataset, perplexity))
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! data_outlier_k_l {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, l, distance))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, l: usize,
            distance: &str,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let result =
                crate::py_interruptible(py, || Ok(outlier::$variant::<_, $dtype>(&dataset, k, l)))?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! data_outlier_n {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, nmin, alpha, g, seed=None, *, distance))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, nmin: usize, alpha: usize,
            g: usize, seed: Option<u64>, distance: &str,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let result = crate::py_interruptible(py, || {
                Ok(outlier::$variant::<_, $dtype>(&dataset, nmin, alpha, g, seed.unwrap_or(0)))
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, distance: &str,
            index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, k)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_m {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, m, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, m: f64, distance: &str,
            index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, k, m)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_alpha {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, alpha, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, alpha: f64,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, k, alpha)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_d {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, d, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, d: $dtype, distance: &str,
            index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, d)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_d_p {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, d, p, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, d: $dtype, p: f64,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, d, p)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_delta {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, delta, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, delta: f64,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, k, delta)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_expect_dist {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, expect, dist, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, expect: f64,
            dist: &str, distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let dist = parse_cop_distance_dist(dist)?;
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, k, expect, dist)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_k_expect_dist!(
    correlation_outlier_probabilities_f32,
    correlation_outlier_probabilities,
    f32
);
tree_outlier_k_expect_dist!(
    correlation_outlier_probabilities_f64,
    correlation_outlier_probabilities,
    f64
);

macro_rules! tree_outlier_k_h_kernel {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, h, kernel, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, h: f64, kernel: &str,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let kernel = parse_kernel_density_function(kernel)?;
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, k, h, kernel)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_k_h_kernel!(simple_kernel_density_lof_f32, simple_kd_lof, f32);
tree_outlier_k_h_kernel!(simple_kernel_density_lof_f64, simple_kd_lof, f64);

// Dispatches an estimator name to a concrete KNNIDEstimator type, forwarding it as
// first argument to the callback macro. Usage:
//   dispatch_id_estimator!(opt_str, callback!(args...))
// The callback must accept ($E:ty, args...).
macro_rules! dispatch_id_estimator {
    ($est:expr, $cb:ident ! ($($args:tt)*)) => {
        match $est.unwrap_or("agg_hill").to_lowercase().as_str() {
            "hill" | "hill_id" => $cb!(HillID, $($args)*),
            "agg_hill" | "aggregated_hill" => $cb!(AggregatedHillID, $($args)*),
            "mom" | "method_of_moments" => $cb!(MethodOfMoments, $($args)*),
            "lmoments" | "lmoments_estimator" => $cb!(LMomentsEstimator, $($args)*),
            "pwm" | "probability_weighted_moments" => $cb!(ProbabilityWeightedMoments, $($args)*),
            "pwm2" | "probability_weighted_moments_2" => $cb!(ProbabilityWeightedMoments2, $($args)*),
            "rv" | "regularly_varying" => $cb!(RVEstimator, $($args)*),
            "ged" | "generalized_expansion_dimension" => $cb!(GeneralizedExpansionDimension, $($args)*),
            "tightlid" | "tight_lid" => $cb!(TightLID, $($args)*),
            "alid" => $cb!(ALID, $($args)*),
            "zipf" | "zipf_id" => $cb!(ZipfID, $($args)*),
            "rabid" => $cb!(RABID, $($args)*),
            "abid" => $cb!(ABID, $($args)*),
            other => return Err(PyValueError::new_err(format!(
                "unknown ID estimator '{}', valid values are: \
                 hill, agg_hill, mom, lmoments, pwm, pwm2, rv, ged, \
                 tightlid, alid, zipf, rabid, abid",
                other
            ))),
        }
    };
}

macro_rules! apply_idos {
    ($E:ty, $tree:expr, $dataset:expr, $kc:expr, $kr:expr) => {
        outlier::intrinsic_dimensionality_outlier_score::<_, _, _, $E>($tree, $dataset, $kc, $kr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?
    };
}

macro_rules! tree_outlier_kc_kr {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k_c, k_r, estimator=None, *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k_c: usize, k_r: usize,
            estimator: Option<&str>, distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = dispatch_id_estimator!(estimator, apply_idos!(tree, &dataset, k_c, k_r));
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_kc_kr!(intrinsic_dimensionality_outlier_score_f32, f32);
tree_outlier_kc_kr!(intrinsic_dimensionality_outlier_score_f64, f64);

macro_rules! apply_lid {
    ($E:ty, $tree:expr, $dataset:expr, $k:expr) => {
        outlier::local_intrinsic_dimensionality::<_, _, _, $E>($tree, $dataset, $k)
    };
}

macro_rules! tree_outlier_k_id {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, estimator=None, *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize,
            estimator: Option<&str>, distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = dispatch_id_estimator!(estimator, apply_lid!(tree, &dataset, k));
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_k_id!(local_intrinsic_dimensionality_f32, f32);
tree_outlier_k_id!(local_intrinsic_dimensionality_f64, f64);

macro_rules! isolation_forest {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, num_trees: usize,
            subsample_size: usize, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDatasetWithDistance::with_distance(&array, Euclidean);
            let result = outlier::isolation_forest::<_, $dtype>(
                &dataset,
                num_trees,
                subsample_size,
                seed.unwrap_or(0),
            );
            result_to_py_outlier(py, result)
        }
    };
}

isolation_forest!(isolation_forest_f32, f32);
isolation_forest!(isolation_forest_f64, f64);

macro_rules! data_outlier_no_args {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, distance))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, distance: &str,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dist_fn = super::parse_distance_fn::<$dtype>(distance)?;
            let dataset =
                NdArrayDatasetWithDistance::<$dtype, _, _>::with_distance(&array, Euclidean);
            let result =
                crate::py_interruptible(py, || Ok(outlier::$variant(&dataset, &*dist_fn)))?;
            result_to_py_outlier(py, result)
        }
    };
    ($name:ident, $variant:ident, $dtype:ty, take_dataset) => {
        #[pyfunction]
        #[pyo3(signature = (data))]
        fn $name<'py>(py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDatasetWithDistance::with_distance(&array, Euclidean);
            let result =
                crate::py_interruptible(py, || Ok(outlier::$variant::<_, $dtype>(dataset)))?;
            result_to_py_outlier(py, result)
        }
    };
    ($name:ident, $variant:ident, $dtype:ty, seed) => {
        #[pyfunction]
        #[pyo3(signature = (data, seed=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, seed: Option<u64>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = NdArrayDatasetWithDistance::with_distance(&array, Euclidean);
            let result = crate::py_interruptible(py, || {
                Ok(outlier::$variant::<_, $dtype>(dataset, seed.unwrap_or(0)))
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

data_outlier_no_args!(zero_f32, zero, f32, take_dataset);
data_outlier_no_args!(zero_f64, zero, f64, take_dataset);

data_outlier_no_args!(random_f32, random, f32, seed);
data_outlier_no_args!(random_f64, random, f64, seed);

data_outlier_no_args!(distance_from_center_f32, distance_from_center, f32);
data_outlier_no_args!(distance_from_center_f64, distance_from_center, f64);

data_outlier_no_args!(distance_from_origin_f32, distance_from_origin, f32);
data_outlier_no_args!(distance_from_origin_f64, distance_from_origin, f64);

macro_rules! tree_outlier_rmax_nmin_alpha {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, rmax, nmin, alpha, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, rmax: $dtype, nmin: usize,
            alpha: $dtype, distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, rmax, nmin, alpha)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_rmax_nmin_alpha!(local_correlation_integral_f32, local_correlation_integral, f32);
tree_outlier_rmax_nmin_alpha!(local_correlation_integral_f64, local_correlation_integral, f64);

// Macro-generated wrappers

tree_outlier_p!(stochastic_outlier_selection_f32, stochastic_outlier_selection, f32);
tree_outlier_p!(stochastic_outlier_selection_f64, stochastic_outlier_selection, f64);

tree_outlier_k_m!(local_outlier_probabilities_f32, local_outlier_probabilities, f32);
tree_outlier_k_m!(local_outlier_probabilities_f64, local_outlier_probabilities, f64);

tree_outlier_d!(db_outlier_score_f32, db_outlier_score, f32);
tree_outlier_d!(db_outlier_score_f64, db_outlier_score, f64);

tree_outlier_d_p!(db_outlier_detection_f32, db_outlier_detection, f32);
tree_outlier_d_p!(db_outlier_detection_f64, db_outlier_detection, f64);

tree_outlier_k_delta!(dynamic_window_outlier_factor_f32, dynamic_window_outlier_factor, f32);
tree_outlier_k_delta!(dynamic_window_outlier_factor_f64, dynamic_window_outlier_factor, f64);

macro_rules! fast_abod {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, kernel="poly2", *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, kernel: &str,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let kfn = parse_abod_kernel::<$dtype>(kernel)?;
            let result = crate::py_interruptible(py, || {
                outlier::fast_angle_based_outlier_detection::<_, _, $dtype, _>(
                    &tree, &dataset, k, kfn,
                )
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

fast_abod!(fast_angle_based_outlier_detection_f32, f32);
fast_abod!(fast_angle_based_outlier_detection_f64, f64);
tree_outlier_k_m!(influence_outlier_f32, influence_outlier, f32);
tree_outlier_k_m!(influence_outlier_f64, influence_outlier, f64);

macro_rules! tree_outlier_k_h_c_kernel {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, h, c, kernel, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, h: f64, c: f64,
            kernel: &str, distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let kernel = parse_kernel_density_function(kernel)?;
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, k, h, c, kernel)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_k_h_c_kernel!(local_density_factor_f32, local_density_factor, f32);
tree_outlier_k_h_c_kernel!(local_density_factor_f64, local_density_factor, f64);

tree_outlier_k!(local_density_outlier_factor_f32, local_density_outlier_factor, f32);
tree_outlier_k!(local_density_outlier_factor_f64, local_density_outlier_factor, f64);

tree_outlier_k!(local_isolation_coefficient_f32, local_isolation_coefficient, f32);
tree_outlier_k!(local_isolation_coefficient_f64, local_isolation_coefficient, f64);

tree_outlier_k!(local_outlier_factor_f32, local_outlier_factor, f32);
tree_outlier_k!(local_outlier_factor_f64, local_outlier_factor, f64);

tree_outlier_k!(
    outlier_detection_independence_neighbor_f32,
    outlier_detection_independence_neighbor,
    f32
);
tree_outlier_k!(
    outlier_detection_independence_neighbor_f64,
    outlier_detection_independence_neighbor,
    f64
);

tree_outlier_k_alpha!(subspace_outlier_degree_f32, subspace_outlier_degree, f32);
tree_outlier_k_alpha!(subspace_outlier_degree_f64, subspace_outlier_degree, f64);

tree_outlier_k!(weighted_knn_f32, weighted_knn, f32);
tree_outlier_k!(weighted_knn_f64, weighted_knn, f64);

macro_rules! tree_outlier_krefer_kreach {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, krefer, kreach, distance, *, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, krefer: usize, kreach: usize,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = crate::py_interruptible(py, || {
                outlier::$variant::<_, _, $dtype>(tree, &dataset, krefer, kreach)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_krefer_kreach!(flexible_lof_f32, flexible_lof, f32);
tree_outlier_krefer_kreach!(flexible_lof_f64, flexible_lof, f64);

tree_outlier_k!(simplified_lof_f32, simplified_lof, f32);
tree_outlier_k!(simplified_lof_f64, simplified_lof, f64);

tree_outlier_k!(k_nearest_neighbors_outlier_f32, k_nearest_neighbors_outlier, f32);
tree_outlier_k!(k_nearest_neighbors_outlier_f64, k_nearest_neighbors_outlier, f64);

tree_outlier_k!(
    k_nearest_neighbors_distance_deviation_f32,
    k_nearest_neighbors_distance_deviation,
    f32
);
tree_outlier_k!(
    k_nearest_neighbors_distance_deviation_f64,
    k_nearest_neighbors_distance_deviation,
    f64
);

tree_outlier_k!(k_nearest_neighbors_sos_f32, k_nearest_neighbors_sos, f32);
tree_outlier_k!(k_nearest_neighbors_sos_f64, k_nearest_neighbors_sos, f64);

macro_rules! abod {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, kernel="poly2", *, distance))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, kernel: &str, distance: &str,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let kfn = parse_abod_kernel::<$dtype>(kernel)?;
            let result = crate::py_interruptible(py, || {
                Ok(outlier::angle_based_outlier_detection::<_, $dtype, _>(&dataset, kfn))
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

abod!(angle_based_outlier_detection_f32, f32);
abod!(angle_based_outlier_detection_f64, f64);

data_outlier_k_l!(lb_abod_f32, lb_abod, f32);
data_outlier_k_l!(lb_abod_f64, lb_abod, f64);

data_outlier_n!(
    approximate_local_correlation_integral_f32,
    approximate_local_correlation_integral,
    f32
);
data_outlier_n!(
    approximate_local_correlation_integral_f64,
    approximate_local_correlation_integral,
    f64
);

// ---- COF -------------------------------------------------------------------

tree_outlier_k!(connectivity_outlier_factor_f32, connectivity_outlier_factor, f32);
tree_outlier_k!(connectivity_outlier_factor_f64, connectivity_outlier_factor, f64);

// ---- VoV -------------------------------------------------------------------

tree_outlier_k!(variance_of_volume_f32, variance_of_volume, f32);
tree_outlier_k!(variance_of_volume_f64, variance_of_volume, f64);

// ---- KDEOS -----------------------------------------------------------------

macro_rules! tree_outlier_kdeos {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, kmin, kmax, kernel="gaussian", min_bandwidth=0.0, scale=1.0, idim=None, *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, kmin: usize, kmax: usize,
            kernel: &str, min_bandwidth: f64, scale: f64, idim: Option<usize>,
            distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let kernel = parse_kernel_density_function(kernel)?;
            let result = crate::py_interruptible(py, || {
                outlier::kdeos::kdeos::<_, _, $dtype>(tree, &dataset, kmin, kmax, kernel, min_bandwidth, scale, idim)
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_kdeos!(kdeos_f32, f32);
tree_outlier_kdeos!(kdeos_f64, f64);

// ---- ISOS ------------------------------------------------------------------

macro_rules! apply_isos {
    ($E:ty, $py:ident, $tree:expr, $dataset:expr, $k:expr) => {
        crate::py_interruptible($py, || {
            outlier::intrinsic_stochastic_outlier_selection::<_, _, _, $E>($tree, $dataset, $k)
        })
    };
}

macro_rules! tree_outlier_isos {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, estimator=None, *, distance, index))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize,
            estimator: Option<&str>, distance: &str, index: PyRef<'_, SearchIndex>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = index.inner();
            let result = dispatch_id_estimator!(estimator, apply_isos!(py, tree, &dataset, k))?;
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_isos!(intrinsic_stochastic_outlier_selection_f32, f32);
tree_outlier_isos!(intrinsic_stochastic_outlier_selection_f64, f64);

// ---- LBABOD kernel variant -------------------------------------------------

macro_rules! lb_abod_kernel {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, l, kernel="poly2", *, distance))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, l: usize, kernel: &str,
            distance: &str,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let kfn = parse_abod_kernel::<$dtype>(kernel)?;
            let result = crate::py_interruptible(py, || {
                Ok(outlier::lb_abod_kernel::<_, $dtype, _>(&dataset, k, l, kfn))
            })?;
            result_to_py_outlier(py, result)
        }
    };
}

lb_abod_kernel!(lb_abod_kernel_f32, f32);
lb_abod_kernel!(lb_abod_kernel_f64, f64);

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(angle_based_outlier_detection_f32))?;
    m.add_wrapped(wrap_pyfunction!(angle_based_outlier_detection_f64))?;
    m.add_wrapped(wrap_pyfunction!(fast_angle_based_outlier_detection_f32))?;
    m.add_wrapped(wrap_pyfunction!(fast_angle_based_outlier_detection_f64))?;
    m.add_wrapped(wrap_pyfunction!(approximate_local_correlation_integral_f32))?;
    m.add_wrapped(wrap_pyfunction!(approximate_local_correlation_integral_f64))?;
    m.add_wrapped(wrap_pyfunction!(distance_from_center_f32))?;
    m.add_wrapped(wrap_pyfunction!(distance_from_center_f64))?;
    m.add_wrapped(wrap_pyfunction!(distance_from_origin_f32))?;
    m.add_wrapped(wrap_pyfunction!(distance_from_origin_f64))?;
    m.add_wrapped(wrap_pyfunction!(random_f32))?;
    m.add_wrapped(wrap_pyfunction!(random_f64))?;
    m.add_wrapped(wrap_pyfunction!(zero_f32))?;
    m.add_wrapped(wrap_pyfunction!(zero_f64))?;
    m.add_wrapped(wrap_pyfunction!(correlation_outlier_probabilities_f32))?;
    m.add_wrapped(wrap_pyfunction!(correlation_outlier_probabilities_f64))?;
    m.add_wrapped(wrap_pyfunction!(db_outlier_detection_f32))?;
    m.add_wrapped(wrap_pyfunction!(db_outlier_detection_f64))?;
    m.add_wrapped(wrap_pyfunction!(db_outlier_score_f32))?;
    m.add_wrapped(wrap_pyfunction!(db_outlier_score_f64))?;
    m.add_wrapped(wrap_pyfunction!(dynamic_window_outlier_factor_f32))?;
    m.add_wrapped(wrap_pyfunction!(dynamic_window_outlier_factor_f64))?;
    m.add_wrapped(wrap_pyfunction!(flexible_lof_f32))?;
    m.add_wrapped(wrap_pyfunction!(flexible_lof_f64))?;
    m.add_wrapped(wrap_pyfunction!(intrinsic_dimensionality_outlier_score_f32))?;
    m.add_wrapped(wrap_pyfunction!(intrinsic_dimensionality_outlier_score_f64))?;
    m.add_wrapped(wrap_pyfunction!(influence_outlier_f32))?;
    m.add_wrapped(wrap_pyfunction!(influence_outlier_f64))?;
    m.add_wrapped(wrap_pyfunction!(isolation_forest_f32))?;
    m.add_wrapped(wrap_pyfunction!(isolation_forest_f64))?;
    m.add_wrapped(wrap_pyfunction!(k_nearest_neighbors_outlier_f32))?;
    m.add_wrapped(wrap_pyfunction!(k_nearest_neighbors_outlier_f64))?;
    m.add_wrapped(wrap_pyfunction!(k_nearest_neighbors_distance_deviation_f32))?;
    m.add_wrapped(wrap_pyfunction!(k_nearest_neighbors_distance_deviation_f64))?;
    m.add_wrapped(wrap_pyfunction!(k_nearest_neighbors_sos_f32))?;
    m.add_wrapped(wrap_pyfunction!(k_nearest_neighbors_sos_f64))?;
    m.add_wrapped(wrap_pyfunction!(local_density_factor_f32))?;
    m.add_wrapped(wrap_pyfunction!(local_density_factor_f64))?;
    m.add_wrapped(wrap_pyfunction!(local_density_outlier_factor_f32))?;
    m.add_wrapped(wrap_pyfunction!(local_density_outlier_factor_f64))?;
    m.add_wrapped(wrap_pyfunction!(local_intrinsic_dimensionality_f32))?;
    m.add_wrapped(wrap_pyfunction!(local_intrinsic_dimensionality_f64))?;
    m.add_wrapped(wrap_pyfunction!(local_isolation_coefficient_f32))?;
    m.add_wrapped(wrap_pyfunction!(local_isolation_coefficient_f64))?;
    m.add_wrapped(wrap_pyfunction!(local_correlation_integral_f32))?;
    m.add_wrapped(wrap_pyfunction!(local_correlation_integral_f64))?;
    m.add_wrapped(wrap_pyfunction!(local_outlier_factor_f32))?;
    m.add_wrapped(wrap_pyfunction!(local_outlier_factor_f64))?;
    m.add_wrapped(wrap_pyfunction!(local_outlier_probabilities_f32))?;
    m.add_wrapped(wrap_pyfunction!(local_outlier_probabilities_f64))?;
    m.add_wrapped(wrap_pyfunction!(lb_abod_f32))?;
    m.add_wrapped(wrap_pyfunction!(lb_abod_f64))?;
    m.add_wrapped(wrap_pyfunction!(outlier_detection_independence_neighbor_f32))?;
    m.add_wrapped(wrap_pyfunction!(outlier_detection_independence_neighbor_f64))?;
    m.add_wrapped(wrap_pyfunction!(simple_kernel_density_lof_f32))?;
    m.add_wrapped(wrap_pyfunction!(simple_kernel_density_lof_f64))?;
    m.add_wrapped(wrap_pyfunction!(simplified_lof_f32))?;
    m.add_wrapped(wrap_pyfunction!(simplified_lof_f64))?;
    m.add_wrapped(wrap_pyfunction!(subspace_outlier_degree_f32))?;
    m.add_wrapped(wrap_pyfunction!(subspace_outlier_degree_f64))?;
    m.add_wrapped(wrap_pyfunction!(stochastic_outlier_selection_f32))?;
    m.add_wrapped(wrap_pyfunction!(stochastic_outlier_selection_f64))?;
    m.add_wrapped(wrap_pyfunction!(weighted_knn_f32))?;
    m.add_wrapped(wrap_pyfunction!(weighted_knn_f64))?;
    m.add_wrapped(wrap_pyfunction!(connectivity_outlier_factor_f32))?;
    m.add_wrapped(wrap_pyfunction!(connectivity_outlier_factor_f64))?;
    m.add_wrapped(wrap_pyfunction!(variance_of_volume_f32))?;
    m.add_wrapped(wrap_pyfunction!(variance_of_volume_f64))?;
    m.add_wrapped(wrap_pyfunction!(kdeos_f32))?;
    m.add_wrapped(wrap_pyfunction!(kdeos_f64))?;
    m.add_wrapped(wrap_pyfunction!(intrinsic_stochastic_outlier_selection_f32))?;
    m.add_wrapped(wrap_pyfunction!(intrinsic_stochastic_outlier_selection_f64))?;
    m.add_wrapped(wrap_pyfunction!(lb_abod_kernel_f32))?;
    m.add_wrapped(wrap_pyfunction!(lb_abod_kernel_f64))?;
    Ok(())
}
