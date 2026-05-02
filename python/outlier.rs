use numpy::{Element, PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::make_rng;
use crate::distance::DistanceFunction;
use crate::intrinsicdimensionality::{
    ABID, ALID, AggregatedHillID, GeneralizedExpansionDimension, HillID, LMomentsEstimator,
    MethodOfMoments, ProbabilityWeightedMoments, ProbabilityWeightedMoments2, RABID, RVEstimator,
    TightLID, ZipfID,
};
use crate::kernel::polynomial::PolynomialKernel;
use crate::outlier::cop::CopDistanceDist;
use crate::outlier::kernel::KernelDensityFunction;
use crate::search::vptree::VPTree;
use crate::{DistanceData, Float, NdArrayDatasetWithDistance, outlier};

fn parse_abod_kernel<F>(kernel: &str) -> PyResult<Box<dyn Fn(&[F], &[F]) -> F + Sync>>
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

fn build_vptree<D, F>(data: &D, seed: Option<u64>) -> VPTree<F>
where
    D: DistanceData<F> + Sync,
    F: Float,
{
    let mut rng = make_rng(seed);
    VPTree::new(data, 5, &mut rng)
}

fn build_outlier_dataset<'a, N>(
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

macro_rules! data_outlier_no_args {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let result = outlier::$variant::<_, $dtype>(&dataset);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_p {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, perplexity, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, perplexity: f64,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, perplexity);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! data_outlier_k_l {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, l, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, l: usize,
            distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let result = outlier::$variant::<_, $dtype>(&dataset, k, l);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! data_outlier_n {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, nmin, alpha, g, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, nmin: usize, alpha: usize,
            g: usize, seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let result =
                outlier::$variant::<_, $dtype>(&dataset, nmin, alpha, g, seed.unwrap_or(0));
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, seed: Option<u64>,
            distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, k);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_m {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, m, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, m: f64,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, k, m);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_alpha {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, alpha, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, alpha: f64,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, k, alpha);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_d {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, d, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, d: $dtype, seed: Option<u64>,
            distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, d);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_d_p {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, d, p, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, d: $dtype, p: f64,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, d, p);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_delta {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, delta, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, delta: f64,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, k, delta);
            result_to_py_outlier(py, result)
        }
    };
}

macro_rules! tree_outlier_k_expect_dist {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, expect, dist, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, expect: f64,
            dist: &str, seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let dist = parse_cop_distance_dist(dist)?;
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, k, expect, dist);
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
        #[pyo3(signature = (data, k, h, kernel, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, h: f64, kernel: &str,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let kernel = parse_kernel_density_function(kernel)?;
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, k, h, kernel);
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
    };
}

macro_rules! tree_outlier_kc_kr {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k_c, k_r, estimator=None, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k_c: usize, k_r: usize,
            estimator: Option<&str>, seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = dispatch_id_estimator!(estimator, apply_idos!(&tree, &dataset, k_c, k_r));
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
        #[pyo3(signature = (data, k, estimator=None, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize,
            estimator: Option<&str>, seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = dispatch_id_estimator!(estimator, apply_lid!(&tree, &dataset, k));
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_k_id!(local_intrinsic_dimensionality_f32, f32);
tree_outlier_k_id!(local_intrinsic_dimensionality_f64, f64);

#[pyfunction]
fn isolation_forest_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, num_trees: usize, subsample_size: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::new(&array);
    let result =
        outlier::isolation_forest::<_, f32>(&dataset, num_trees, subsample_size, seed.unwrap_or(0));
    result_to_py_outlier(py, result)
}

#[pyfunction]
fn isolation_forest_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, num_trees: usize, subsample_size: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::new(&array);
    let result =
        outlier::isolation_forest::<_, f64>(&dataset, num_trees, subsample_size, seed.unwrap_or(0));
    result_to_py_outlier(py, result)
}

#[pyfunction]
fn zero_f32<'py>(py: Python<'py>, data: PyReadonlyArray2<'py, f32>) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::new(&array);
    let result = outlier::zero::<_, f32>(dataset);
    result_to_py_outlier(py, result)
}

#[pyfunction]
fn zero_f64<'py>(py: Python<'py>, data: PyReadonlyArray2<'py, f64>) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::new(&array);
    let result = outlier::zero::<_, f64>(dataset);
    result_to_py_outlier(py, result)
}

#[pyfunction]
fn random_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::new(&array);
    let result = outlier::random::<_, f32>(dataset, seed.unwrap_or(0));
    result_to_py_outlier(py, result)
}

#[pyfunction]
fn random_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = NdArrayDatasetWithDistance::new(&array);
    let result = outlier::random::<_, f64>(dataset, seed.unwrap_or(0));
    result_to_py_outlier(py, result)
}

macro_rules! tree_outlier_rmax_nmin_alpha {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, rmax, nmin, alpha, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, rmax: $dtype, nmin: usize,
            alpha: $dtype, seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, rmax, nmin, alpha);
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

#[pyfunction]
#[pyo3(signature = (data, k, kernel="poly2", seed=None, distance=None))]
fn fast_angle_based_outlier_detection_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, k: usize, kernel: &str, seed: Option<u64>,
    distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_outlier_dataset::<f32>(&array, distance)?;
    let tree = build_vptree(&dataset, seed);
    let kfn = parse_abod_kernel::<f32>(kernel)?;
    let result =
        outlier::fast_angle_based_outlier_detection::<_, _, f32, _>(&tree, &dataset, k, kfn);
    result_to_py_outlier(py, result)
}

#[pyfunction]
#[pyo3(signature = (data, k, kernel="poly2", seed=None, distance=None))]
fn fast_angle_based_outlier_detection_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, k: usize, kernel: &str, seed: Option<u64>,
    distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_outlier_dataset::<f64>(&array, distance)?;
    let tree = build_vptree(&dataset, seed);
    let kfn = parse_abod_kernel::<f64>(kernel)?;
    let result =
        outlier::fast_angle_based_outlier_detection::<_, _, f64, _>(&tree, &dataset, k, kfn);
    result_to_py_outlier(py, result)
}
tree_outlier_k_m!(influence_outlier_f32, influence_outlier, f32);
tree_outlier_k_m!(influence_outlier_f64, influence_outlier, f64);

macro_rules! tree_outlier_k_h_c_kernel {
    ($name:ident, $variant:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, h, c, kernel, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize, h: f64, c: f64,
            kernel: &str, seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let kernel = parse_kernel_density_function(kernel)?;
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, k, h, c, kernel);
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
        #[pyo3(signature = (data, krefer, kreach, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, krefer: usize, kreach: usize,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = outlier::$variant::<_, _, $dtype>(&tree, &dataset, krefer, kreach);
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

#[pyfunction]
#[pyo3(signature = (data, kernel="poly2", distance=None))]
fn angle_based_outlier_detection_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, kernel: &str, distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_outlier_dataset::<f32>(&array, distance)?;
    let kfn = parse_abod_kernel::<f32>(kernel)?;
    let result = outlier::angle_based_outlier_detection::<_, f32, _>(&dataset, kfn);
    result_to_py_outlier(py, result)
}

#[pyfunction]
#[pyo3(signature = (data, kernel="poly2", distance=None))]
fn angle_based_outlier_detection_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, kernel: &str, distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_outlier_dataset::<f64>(&array, distance)?;
    let kfn = parse_abod_kernel::<f64>(kernel)?;
    let result = outlier::angle_based_outlier_detection::<_, f64, _>(&dataset, kfn);
    result_to_py_outlier(py, result)
}

data_outlier_no_args!(distance_from_center_f32, distance_from_center, f32);
data_outlier_no_args!(distance_from_center_f64, distance_from_center, f64);

data_outlier_no_args!(distance_from_origin_f32, distance_from_origin, f32);
data_outlier_no_args!(distance_from_origin_f64, distance_from_origin, f64);

data_outlier_k_l!(locality_based_abod_f32, locality_based_abod, f32);
data_outlier_k_l!(locality_based_abod_f64, locality_based_abod, f64);

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
        #[pyo3(signature = (data, kmin, kmax, kernel="gaussian", min_bandwidth=0.0, scale=1.0, idim=None, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, kmin: usize, kmax: usize,
            kernel: &str, min_bandwidth: f64, scale: f64, idim: Option<usize>,
            seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let kernel = parse_kernel_density_function(kernel)?;
            let result =
                outlier::kdeos::kdeos::<_, _, $dtype>(&tree, &dataset, kmin, kmax, kernel, min_bandwidth, scale, idim);
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_kdeos!(kdeos_f32, f32);
tree_outlier_kdeos!(kdeos_f64, f64);

// ---- ISOS ------------------------------------------------------------------

macro_rules! apply_isos {
    ($E:ty, $tree:expr, $dataset:expr, $k:expr) => {
        outlier::intrinsic_stochastic_outlier_selection::<_, _, _, $E>($tree, $dataset, $k)
    };
}

macro_rules! tree_outlier_isos {
    ($name:ident, $dtype:ty) => {
        #[pyfunction]
        #[pyo3(signature = (data, k, estimator=None, seed=None, distance=None))]
        fn $name<'py>(
            py: Python<'py>, data: PyReadonlyArray2<'py, $dtype>, k: usize,
            estimator: Option<&str>, seed: Option<u64>, distance: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            let array = data.as_array();
            let dataset = build_outlier_dataset::<$dtype>(&array, distance)?;
            let tree = build_vptree(&dataset, seed);
            let result = dispatch_id_estimator!(estimator, apply_isos!(&tree, &dataset, k));
            result_to_py_outlier(py, result)
        }
    };
}

tree_outlier_isos!(intrinsic_stochastic_outlier_selection_f32, f32);
tree_outlier_isos!(intrinsic_stochastic_outlier_selection_f64, f64);

// ---- LBABOD kernel variant -------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, k, l, kernel="poly2", distance=None))]
fn locality_based_abod_kernel_f32<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f32>, k: usize, l: usize, kernel: &str,
    distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_outlier_dataset::<f32>(&array, distance)?;
    let kfn = parse_abod_kernel::<f32>(kernel)?;
    let result = outlier::locality_based_abod_kernel::<_, f32, _>(&dataset, k, l, kfn);
    result_to_py_outlier(py, result)
}

#[pyfunction]
#[pyo3(signature = (data, k, l, kernel="poly2", distance=None))]
fn locality_based_abod_kernel_f64<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'py, f64>, k: usize, l: usize, kernel: &str,
    distance: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let array = data.as_array();
    let dataset = build_outlier_dataset::<f64>(&array, distance)?;
    let kfn = parse_abod_kernel::<f64>(kernel)?;
    let result = outlier::locality_based_abod_kernel::<_, f64, _>(&dataset, k, l, kfn);
    result_to_py_outlier(py, result)
}

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
    m.add_wrapped(wrap_pyfunction!(locality_based_abod_f32))?;
    m.add_wrapped(wrap_pyfunction!(locality_based_abod_f64))?;
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
    m.add_wrapped(wrap_pyfunction!(locality_based_abod_kernel_f32))?;
    m.add_wrapped(wrap_pyfunction!(locality_based_abod_kernel_f64))?;
    Ok(())
}
