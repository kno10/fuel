use ndarray::ArrayView2;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::cluster::hierarchical::MergeHistory;
use crate::evaluation::cluster::external::contingency_table::ClusterContingencyTable;
use crate::evaluation::cluster::external::{
    BCubed, Entropy, MaximumMatchingAccuracy, PairCounting, PairSetsIndex, SetMatchingPurity,
};
use crate::evaluation::cluster::internal::c_index::c_index;
use crate::evaluation::cluster::internal::cluster_radius::cluster_radius;
use crate::evaluation::cluster::internal::concordance::concordant_pairs_gamma_tau;
use crate::evaluation::cluster::internal::cophenetic::{
    cophenetic_correlation, cophenetic_distances,
};
use crate::evaluation::cluster::internal::davies_bouldin::davies_bouldin_index;
use crate::evaluation::cluster::internal::dbcv::dbcv;
use crate::evaluation::cluster::internal::helpers::NoiseHandling;
use crate::evaluation::cluster::internal::neighbor_consistency::neighbor_consistency_knn;
use crate::evaluation::cluster::internal::pbm_index::pbm_index;
use crate::evaluation::cluster::internal::silhouette::{silhouette, simplified_silhouette};
use crate::evaluation::cluster::internal::squared_errors::squared_errors;
use crate::evaluation::cluster::internal::variance_ratio::variance_ratio_criterion;
use crate::evaluation::outlier::average_precision::average_precision;
use crate::evaluation::outlier::discounted_cumulative_gain::{
    dcg, normalized_discounted_cumulative_gain,
};
use crate::evaluation::outlier::maximum_f1::maximum_f1;
use crate::evaluation::outlier::precision_at_k::{precision_at_k, r_precision};
use crate::evaluation::outlier::precision_recall_curve::{auprc, pr_curve};
use crate::evaluation::outlier::precision_recall_gain::prg_auc;
use crate::evaluation::outlier::receiver_operating_curve::auc;

// ---- helpers ---------------------------------------------------------------

fn labels_from_array(arr: &PyReadonlyArray1<'_, i64>) -> Vec<isize> {
    arr.as_array().iter().map(|&v| v as isize).collect()
}

fn data_from_array(arr: &ArrayView2<'_, f64>) -> Vec<Vec<f64>> {
    (0..arr.nrows()).map(|i| arr.row(i).to_vec()).collect()
}

fn parse_noise_handling(s: &str) -> PyResult<NoiseHandling> {
    match s.to_lowercase().as_str() {
        "ignore" | "ignore_noise" => Ok(NoiseHandling::IgnoreNoise),
        "singletons" | "treat_noise_as_singletons" => Ok(NoiseHandling::TreatNoiseAsSingletons),
        "merge" | "merge_noise" => Ok(NoiseHandling::MergeNoise),
        other => Err(PyValueError::new_err(format!(
            "unknown noise_handling '{}', valid: ignore, singletons, merge",
            other
        ))),
    }
}

/// Convert a scipy-style linkage matrix (n-1, 4) into a MergeHistory<f64>.
/// Columns: idx1, idx2, distance, size.
fn linkage_to_merge_history(arr: &ArrayView2<'_, f64>) -> MergeHistory<f64> {
    let n_merges = arr.nrows();
    let mut idx1 = Vec::with_capacity(n_merges);
    let mut idx2 = Vec::with_capacity(n_merges);
    let mut distance = Vec::with_capacity(n_merges);
    let mut size = Vec::with_capacity(n_merges);

    for i in 0..n_merges {
        idx1.push(arr[(i, 0)] as usize);
        idx2.push(arr[(i, 1)] as usize);
        distance.push(arr[(i, 2)]);
        size.push(arr[(i, 3)] as usize);
    }

    MergeHistory { idx1, idx2, distance, size, prototype: None }
}

// ---- external: pair counting -----------------------------------------------

#[pyfunction]
#[pyo3(signature = (labels1, labels2, self_pairing=false, break_noise_clusters=false,
                    noise_label1=None, noise_label2=None))]
fn pair_counting<'py>(
    py: Python<'py>, labels1: PyReadonlyArray1<'_, i64>, labels2: PyReadonlyArray1<'_, i64>,
    self_pairing: bool, break_noise_clusters: bool, noise_label1: Option<i64>,
    noise_label2: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let l1 = labels_from_array(&labels1);
    let l2 = labels_from_array(&labels2);
    let table = ClusterContingencyTable::from_labels(
        &l1,
        &l2,
        self_pairing,
        break_noise_clusters,
        noise_label1.map(|v| v as isize),
        noise_label2.map(|v| v as isize),
    );
    let pc = PairCounting::new(&table);
    let d = PyDict::new(py);
    d.set_item("in_both", pc.in_both)?;
    d.set_item("in_first", pc.in_first)?;
    d.set_item("in_second", pc.in_second)?;
    d.set_item("in_none", pc.in_none)?;
    d.set_item("f1", pc.f1_measure())?;
    d.set_item("precision", pc.precision())?;
    d.set_item("recall", pc.recall())?;
    d.set_item("fowlkes_mallows", pc.fowlkes_mallows())?;
    d.set_item("rand_index", pc.rand_index())?;
    d.set_item("adjusted_rand_index", pc.adjusted_rand_index())?;
    d.set_item("jaccard", pc.jaccard())?;
    d.set_item("mirkin", pc.mirkin())?;
    d.into_py_any(py)
}

// ---- external: entropy / mutual information --------------------------------

#[pyfunction]
#[pyo3(signature = (labels1, labels2, self_pairing=false, break_noise_clusters=false,
                    noise_label1=None, noise_label2=None))]
fn entropy_measures<'py>(
    py: Python<'py>, labels1: PyReadonlyArray1<'_, i64>, labels2: PyReadonlyArray1<'_, i64>,
    self_pairing: bool, break_noise_clusters: bool, noise_label1: Option<i64>,
    noise_label2: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let l1 = labels_from_array(&labels1);
    let l2 = labels_from_array(&labels2);
    let table = ClusterContingencyTable::from_labels(
        &l1,
        &l2,
        self_pairing,
        break_noise_clusters,
        noise_label1.map(|v| v as isize),
        noise_label2.map(|v| v as isize),
    );
    let e = Entropy::new(&table);
    let d = PyDict::new(py);
    d.set_item("entropy_first", e.entropy_first)?;
    d.set_item("entropy_second", e.entropy_second)?;
    d.set_item("entropy_joint", e.entropy_joint)?;
    d.set_item("mutual_information", e.mutual_information)?;
    d.set_item("variation_of_information", e.variation_of_information)?;
    d.set_item("expected_mutual_information", e.expected_mutual_information)?;
    d.set_item("conditional_entropy_first", e.conditional_entropy_first())?;
    d.set_item("conditional_entropy_second", e.conditional_entropy_second())?;
    d.set_item("upper_bound_mi", e.upper_bound_mi())?;
    d.set_item("upper_bound_vi", e.upper_bound_vi())?;
    d.set_item("joint_nmi", e.joint_nmi())?;
    d.set_item("min_nmi", e.min_nmi())?;
    d.set_item("max_nmi", e.max_nmi())?;
    d.set_item("arithmetic_nmi", e.arithmetic_nmi())?;
    d.set_item("geometric_nmi", e.geometric_nmi())?;
    d.set_item("normalized_vi", e.normalized_variation_of_information())?;
    d.into_py_any(py)
}

// ---- external: BCubed ------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (labels1, labels2, self_pairing=false, break_noise_clusters=false,
                    noise_label1=None, noise_label2=None))]
fn bcubed<'py>(
    py: Python<'py>, labels1: PyReadonlyArray1<'_, i64>, labels2: PyReadonlyArray1<'_, i64>,
    self_pairing: bool, break_noise_clusters: bool, noise_label1: Option<i64>,
    noise_label2: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let l1 = labels_from_array(&labels1);
    let l2 = labels_from_array(&labels2);
    let table = ClusterContingencyTable::from_labels(
        &l1,
        &l2,
        self_pairing,
        break_noise_clusters,
        noise_label1.map(|v| v as isize),
        noise_label2.map(|v| v as isize),
    );
    let bc = BCubed::new(&table);
    let d = PyDict::new(py);
    d.set_item("precision", bc.precision)?;
    d.set_item("recall", bc.recall)?;
    d.set_item("f1", bc.f1_measure())?;
    d.into_py_any(py)
}

// ---- external: set matching purity -----------------------------------------

#[pyfunction]
#[pyo3(signature = (labels1, labels2, self_pairing=false, break_noise_clusters=false,
                    noise_label1=None, noise_label2=None))]
fn set_matching_purity<'py>(
    py: Python<'py>, labels1: PyReadonlyArray1<'_, i64>, labels2: PyReadonlyArray1<'_, i64>,
    self_pairing: bool, break_noise_clusters: bool, noise_label1: Option<i64>,
    noise_label2: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let l1 = labels_from_array(&labels1);
    let l2 = labels_from_array(&labels2);
    let table = ClusterContingencyTable::from_labels(
        &l1,
        &l2,
        self_pairing,
        break_noise_clusters,
        noise_label1.map(|v| v as isize),
        noise_label2.map(|v| v as isize),
    );
    let smp = SetMatchingPurity::new(&table);
    let d = PyDict::new(py);
    d.set_item("purity", smp.purity)?;
    d.set_item("inverse_purity", smp.inverse_purity)?;
    d.set_item("f_first", smp.f_first)?;
    d.set_item("f_second", smp.f_second)?;
    d.into_py_any(py)
}

// ---- external: maximum matching accuracy -----------------------------------

#[pyfunction]
#[pyo3(signature = (labels1, labels2, self_pairing=false, break_noise_clusters=false,
                    noise_label1=None, noise_label2=None))]
fn maximum_matching_accuracy<'py>(
    _py: Python<'py>, labels1: PyReadonlyArray1<'_, i64>, labels2: PyReadonlyArray1<'_, i64>,
    self_pairing: bool, break_noise_clusters: bool, noise_label1: Option<i64>,
    noise_label2: Option<i64>,
) -> PyResult<f64> {
    let l1 = labels_from_array(&labels1);
    let l2 = labels_from_array(&labels2);
    let table = ClusterContingencyTable::from_labels(
        &l1,
        &l2,
        self_pairing,
        break_noise_clusters,
        noise_label1.map(|v| v as isize),
        noise_label2.map(|v| v as isize),
    );
    Ok(MaximumMatchingAccuracy::new(&table).accuracy)
}

// ---- external: pair sets index ---------------------------------------------

#[pyfunction]
#[pyo3(signature = (labels1, labels2, self_pairing=false, break_noise_clusters=false,
                    noise_label1=None, noise_label2=None))]
fn pair_sets_index<'py>(
    py: Python<'py>, labels1: PyReadonlyArray1<'_, i64>, labels2: PyReadonlyArray1<'_, i64>,
    self_pairing: bool, break_noise_clusters: bool, noise_label1: Option<i64>,
    noise_label2: Option<i64>,
) -> PyResult<Py<PyAny>> {
    let l1 = labels_from_array(&labels1);
    let l2 = labels_from_array(&labels2);
    let table = ClusterContingencyTable::from_labels(
        &l1,
        &l2,
        self_pairing,
        break_noise_clusters,
        noise_label1.map(|v| v as isize),
        noise_label2.map(|v| v as isize),
    );
    let psi = PairSetsIndex::new(&table);
    let d = PyDict::new(py);
    d.set_item("simplified_psi", psi.simplified_psi)?;
    d.set_item("psi", psi.psi)?;
    d.into_py_any(py)
}

// ---- internal: silhouette --------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore", penalize=false))]
fn simplified_silhouette_score<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str, penalize: bool,
) -> PyResult<Py<PyAny>> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    let res = simplified_silhouette(&d, &l, noise_label.map(|v| v as isize), nh, penalize);
    let out = PyDict::new(py);
    out.set_item("mean", res.mean)?;
    out.set_item("stddev", res.stddev)?;
    out.set_item("values", PyArray1::from_vec(py, res.values).into_pyobject(py)?.into_any())?;
    out.into_py_any(py)
}

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore", penalize=false))]
fn silhouette_score<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str, penalize: bool,
) -> PyResult<Py<PyAny>> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    let res = silhouette(&d, &l, noise_label.map(|v| v as isize), nh, penalize);
    let out = PyDict::new(py);
    out.set_item("mean", res.mean)?;
    out.set_item("stddev", res.stddev)?;
    out.set_item("values", PyArray1::from_vec(py, res.values).into_pyobject(py)?.into_any())?;
    out.into_py_any(py)
}

// ---- internal: Davies-Bouldin ----------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore", p=1.0))]
fn davies_bouldin<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str, p: f64,
) -> PyResult<f64> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    Ok(davies_bouldin_index(&d, &l, noise_label.map(|v| v as isize), nh, p))
}

// ---- internal: variance ratio (Calinski-Harabasz) --------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore", penalize=false))]
fn variance_ratio<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str, penalize: bool,
) -> PyResult<f64> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    Ok(variance_ratio_criterion(&d, &l, noise_label.map(|v| v as isize), nh, penalize))
}

// ---- internal: squared errors (SSE/RMSD) -----------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore"))]
fn squared_error_stats<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str,
) -> PyResult<Py<PyAny>> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    let res = squared_errors(&d, &l, noise_label.map(|v| v as isize), nh);
    let out = PyDict::new(py);
    out.set_item("mean", res.mean)?;
    out.set_item("sum_of_squares", res.sum_of_squares)?;
    out.set_item("rmsd", res.rmsd)?;
    out.into_py_any(py)
}

// ---- internal: c-index -----------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore"))]
fn c_index_score<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str,
) -> PyResult<f64> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    Ok(c_index(&d, &l, noise_label.map(|v| v as isize), nh))
}

// ---- internal: concordance (gamma / tau) -----------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore"))]
fn concordance<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str,
) -> PyResult<Py<PyAny>> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    let res = concordant_pairs_gamma_tau(&d, &l, noise_label.map(|v| v as isize), nh);
    let out = PyDict::new(py);
    out.set_item("gamma", res.gamma)?;
    out.set_item("tau", res.tau)?;
    out.into_py_any(py)
}

// ---- internal: cluster radius ----------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore"))]
fn cluster_radius_stats<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str,
) -> PyResult<Py<PyAny>> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    let res = cluster_radius(&d, &l, noise_label.map(|v| v as isize), nh);
    let out = PyDict::new(py);
    out.set_item("weighted", res.weighted)?;
    out.set_item("unweighted", res.unweighted)?;
    out.into_py_any(py)
}

// ---- internal: neighbor consistency ----------------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, k))]
fn neighbor_consistency<'py>(
    py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>, k: usize,
) -> PyResult<Py<PyAny>> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let res = neighbor_consistency_knn(&d, &l, k);
    let out = PyDict::new(py);
    out.set_item("average", res.average)?;
    out.set_item("full", res.full)?;
    out.set_item(
        "per_element_average",
        PyArray1::from_vec(py, res.per_element_average).into_pyobject(py)?.into_any(),
    )?;
    out.set_item(
        "per_element_full",
        PyArray1::from_vec(py, res.per_element_full).into_pyobject(py)?.into_any(),
    )?;
    out.into_py_any(py)
}

// ---- internal: PBM index ---------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None, noise_handling="ignore"))]
fn pbm<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>, noise_handling: &str,
) -> PyResult<f64> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    let nh = parse_noise_handling(noise_handling)?;
    Ok(pbm_index(&d, &l, noise_label.map(|v| v as isize), nh))
}

// ---- internal: DBCV --------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (data, labels, noise_label=None))]
fn dbcv_score<'py>(
    _py: Python<'py>, data: PyReadonlyArray2<'_, f64>, labels: PyReadonlyArray1<'_, i64>,
    noise_label: Option<i64>,
) -> PyResult<f64> {
    let arr = data.as_array();
    let d = data_from_array(&arr);
    let l: Vec<isize> = labels.as_array().iter().map(|&v| v as isize).collect();
    Ok(dbcv(&d, &l, noise_label.map(|v| v as isize)))
}

// ---- cophenetic distances / correlation ------------------------------------

/// Compute pairwise cophenetic distances from a scipy-style linkage matrix.
/// `linkage` is (n-1, 4): [idx1, idx2, distance, size].
/// Returns a condensed lower-triangular distance vector of length n*(n-1)/2.
#[pyfunction]
fn cophenetic_distance_vector<'py>(
    py: Python<'py>, linkage: PyReadonlyArray2<'_, f64>,
) -> PyResult<Py<PyAny>> {
    let arr = linkage.as_array();
    let n_merges = arr.nrows();
    let n = n_merges + 1;
    let history = linkage_to_merge_history(&arr);
    let v = cophenetic_distances(&history, n);
    PyArray1::from_vec(py, v).into_py_any(py)
}

/// Compute Pearson correlation between the cophenetic distances of two
/// scipy-style linkage matrices (each (n-1, 4)).
#[pyfunction]
fn cophenetic_corr<'py>(
    _py: Python<'py>, linkage1: PyReadonlyArray2<'_, f64>, linkage2: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    let arr1 = linkage1.as_array();
    let arr2 = linkage2.as_array();
    if arr1.nrows() != arr2.nrows() {
        return Err(PyValueError::new_err("linkage matrices must have the same number of rows"));
    }
    let n = arr1.nrows() + 1;
    let h1 = linkage_to_merge_history(&arr1);
    let h2 = linkage_to_merge_history(&arr2);
    Ok(cophenetic_correlation(&h1, &h2, n))
}

// ---- outlier evaluation ----------------------------------------------------

fn scores_from_array(arr: &PyReadonlyArray1<'_, f64>) -> Vec<f64> { arr.as_array().to_vec() }

fn binary_labels_from_array(arr: &PyReadonlyArray1<'_, u8>) -> Vec<u8> { arr.as_array().to_vec() }

#[pyfunction]
fn outlier_auc(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(auc(&scores_from_array(&scores), &binary_labels_from_array(&labels)))
}

#[pyfunction]
fn outlier_average_precision(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(average_precision(&scores_from_array(&scores), &binary_labels_from_array(&labels)))
}

#[pyfunction]
fn outlier_auprc(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(auprc(&scores_from_array(&scores), &binary_labels_from_array(&labels)))
}

#[pyfunction]
fn outlier_pr_curve<'py>(
    py: Python<'py>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<Py<PyAny>> {
    let curve = pr_curve(&scores_from_array(&scores), &binary_labels_from_array(&labels));
    let recall: Vec<f64> = curve.iter().map(|&(r, _)| r).collect();
    let precision: Vec<f64> = curve.iter().map(|&(_, p)| p).collect();
    let out = PyDict::new(py);
    out.set_item("recall", PyArray1::from_vec(py, recall).into_pyobject(py)?.into_any())?;
    out.set_item("precision", PyArray1::from_vec(py, precision).into_pyobject(py)?.into_any())?;
    out.into_py_any(py)
}

#[pyfunction]
fn outlier_prg_auc(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(prg_auc(&scores_from_array(&scores), &binary_labels_from_array(&labels)))
}

#[pyfunction]
fn outlier_dcg(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(dcg(&scores_from_array(&scores), &binary_labels_from_array(&labels)))
}

#[pyfunction]
fn outlier_ndcg(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(normalized_discounted_cumulative_gain(
        &scores_from_array(&scores),
        &binary_labels_from_array(&labels),
    ))
}

#[pyfunction]
fn outlier_maximum_f1(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(maximum_f1(&scores_from_array(&scores), &binary_labels_from_array(&labels)))
}

#[pyfunction]
fn outlier_precision_at_k(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>, k: usize,
) -> PyResult<f64> {
    Ok(precision_at_k(&scores_from_array(&scores), &binary_labels_from_array(&labels), k))
}

#[pyfunction]
fn outlier_r_precision(
    _py: Python<'_>, scores: PyReadonlyArray1<'_, f64>, labels: PyReadonlyArray1<'_, u8>,
) -> PyResult<f64> {
    Ok(r_precision(&scores_from_array(&scores), &binary_labels_from_array(&labels)))
}

// ---- module registration ---------------------------------------------------

pub fn register<'py>(m: &'py Bound<'py, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(pair_counting))?;
    m.add_wrapped(wrap_pyfunction!(entropy_measures))?;
    m.add_wrapped(wrap_pyfunction!(bcubed))?;
    m.add_wrapped(wrap_pyfunction!(set_matching_purity))?;
    m.add_wrapped(wrap_pyfunction!(maximum_matching_accuracy))?;
    m.add_wrapped(wrap_pyfunction!(pair_sets_index))?;
    m.add_wrapped(wrap_pyfunction!(simplified_silhouette_score))?;
    m.add_wrapped(wrap_pyfunction!(silhouette_score))?;
    m.add_wrapped(wrap_pyfunction!(davies_bouldin))?;
    m.add_wrapped(wrap_pyfunction!(variance_ratio))?;
    m.add_wrapped(wrap_pyfunction!(squared_error_stats))?;
    m.add_wrapped(wrap_pyfunction!(c_index_score))?;
    m.add_wrapped(wrap_pyfunction!(concordance))?;
    m.add_wrapped(wrap_pyfunction!(cluster_radius_stats))?;
    m.add_wrapped(wrap_pyfunction!(neighbor_consistency))?;
    m.add_wrapped(wrap_pyfunction!(pbm))?;
    m.add_wrapped(wrap_pyfunction!(dbcv_score))?;
    m.add_wrapped(wrap_pyfunction!(cophenetic_distance_vector))?;
    m.add_wrapped(wrap_pyfunction!(cophenetic_corr))?;
    m.add_wrapped(wrap_pyfunction!(outlier_auc))?;
    m.add_wrapped(wrap_pyfunction!(outlier_average_precision))?;
    m.add_wrapped(wrap_pyfunction!(outlier_auprc))?;
    m.add_wrapped(wrap_pyfunction!(outlier_pr_curve))?;
    m.add_wrapped(wrap_pyfunction!(outlier_prg_auc))?;
    m.add_wrapped(wrap_pyfunction!(outlier_dcg))?;
    m.add_wrapped(wrap_pyfunction!(outlier_ndcg))?;
    m.add_wrapped(wrap_pyfunction!(outlier_maximum_f1))?;
    m.add_wrapped(wrap_pyfunction!(outlier_precision_at_k))?;
    m.add_wrapped(wrap_pyfunction!(outlier_r_precision))?;
    Ok(())
}
