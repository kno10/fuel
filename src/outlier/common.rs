use std::collections::HashMap;

use num_traits::ToPrimitive;

use crate::{DistanceData, Float, IndexQuery, KnnSearch, RangeSearch};

/// Iterate over all points in `data`, perform kNN search with `tree` and invoke
/// `op` with each index and the filtered neighbor list (excluding self).
///
/// The operation returns a value per point, the returned vector preserves the
/// original order of indices and is sorted by `idx` semantics.
#[cfg(feature = "parallel")]
pub fn for_each_knn<'a, S, D, F, R, Op>(
    tree: &S, data: &'a D, k_effective: usize, include_self: bool, op: Op,
) -> Vec<R>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    Op: Fn(usize, Vec<(usize, F)>) -> R + Send + Sync,
    R: Send,
{
    use rayon::prelude::*;

    let k_query = k_effective + if include_self { 0 } else { 1 };
    (0..data.size())
        .into_par_iter()
        .map(|idx| {
            let mut query = data.query();
            query.set_index(idx);
            let mut neighbors: Vec<(usize, F)> = tree
                .search_knn(&query, k_query)
                .into_iter()
                .map(|neighbor| (neighbor.index, neighbor.distance))
                .collect();
            if !include_self {
                neighbors.retain(|neighbor| neighbor.0 != idx);
                neighbors.truncate(k_effective);
            }
            op(idx, neighbors)
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub fn for_each_knn<'a, S, D, F, R, Op>(
    tree: &S, data: &'a D, k_effective: usize, include_self: bool, op: Op,
) -> Vec<R>
where
    F: Float,
    D: DistanceData<F> + 'a,
    S: KnnSearch<F, D::Query<'a>>,
    Op: Fn(usize, Vec<(usize, F)>) -> R,
{
    let k_query = k_effective + if include_self { 0 } else { 1 };
    data.iter()
        .map(|idx| {
            let mut query = data.query();
            query.set_index(idx);

            let mut neighbors: Vec<(usize, F)> = tree
                .search_knn(&query, k_query)
                .into_iter()
                .map(|neighbor| (neighbor.index, neighbor.distance))
                .collect();
            if !include_self {
                neighbors.retain(|neighbor| neighbor.0 != idx);
                neighbors.truncate(k_effective);
            }
            op(idx, neighbors)
        })
        .collect()
}

#[cfg(feature = "parallel")]
pub fn for_each_range<'a, S, D, F, R, Op>(
    tree: &S, data: &'a D, d: F, include_self: bool, op: Op,
) -> Vec<R>
where
    F: Float + Send + Sync,
    D: DistanceData<F> + Sync + 'a,
    S: RangeSearch<F, D::Query<'a>> + Sync,
    Op: Fn(usize, Vec<(usize, F)>) -> R + Send + Sync,
    R: Send,
{
    use rayon::prelude::*;

    (0..data.size())
        .into_par_iter()
        .map(|idx| {
            let mut query = data.query();
            query.set_index(idx);
            let neighbors: Vec<(usize, F)> = tree
                .search_range(&query, d)
                .into_iter()
                .filter(|neighbor| include_self || neighbor.index != idx)
                .map(|neighbor| (neighbor.index, neighbor.distance))
                .collect();
            op(idx, neighbors)
        })
        .collect()
}

#[cfg(not(feature = "parallel"))]
pub fn for_each_range<'a, S, D, F, R, Op>(
    tree: &S, data: &'a D, d: F, include_self: bool, op: Op,
) -> Vec<R>
where
    F: Float,
    D: DistanceData<F> + 'a,
    S: RangeSearch<F, D::Query<'a>>,
    Op: Fn(usize, Vec<(usize, F)>) -> R,
{
    data.iter()
        .map(|idx| {
            let mut query = data.query();
            query.set_index(idx);
            let neighbors: Vec<(usize, F)> = tree
                .search_range(&query, d)
                .into_iter()
                .filter(|neighbor| include_self || neighbor.index != idx)
                .map(|neighbor| (neighbor.index, neighbor.distance))
                .collect();
            op(idx, neighbors)
        })
        .collect()
}

/// A minimal trait used by helper utilities that operate on outlier scores.
#[derive(Debug, Clone, PartialEq)]
pub struct OutlierMetadata<F: Float> {
    pub label: String,
    pub ascending: bool,
    pub baseline: F,
    pub minimum: F,
    pub maximum: F,
    pub theoretical_minimum: F,
    pub theoretical_maximum: F,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutlierResult<F: Float> {
    pub scores: Vec<F>,
    pub metadata: OutlierMetadata<F>,
}

fn compute_min_max<F: Float>(scores: &[F]) -> (F, F) {
    let mut min = F::infinity();
    let mut max = F::neg_infinity();
    for &v in scores {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if min.is_infinite() && max.is_infinite() {
        min = F::zero();
        max = F::zero();
    }
    (min, max)
}

pub fn make_outlier_result<F: Float>(
    scores: Vec<F>, label: &str, ascending: bool, baseline: F, theoretical_min: F,
    theoretical_max: F,
) -> OutlierResult<F> {
    let (minimum, maximum) = compute_min_max(&scores);
    OutlierResult {
        scores,
        metadata: OutlierMetadata {
            label: label.to_string(),
            ascending,
            baseline,
            minimum,
            maximum,
            theoretical_minimum: theoretical_min,
            theoretical_maximum: theoretical_max,
        },
    }
}

use std::fs;
use std::path::PathBuf;

pub fn load_gaussian4d_points() -> Vec<Vec<f64>> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("testdata");
    path.push("6-gaussian-4d.csv");

    let contents = fs::read_to_string(&path).expect("Failed to read gaussian testdata");
    let mut points = Vec::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        let point: Vec<f64> = parts[0..4]
            .iter()
            .map(|x| x.parse::<f64>().expect("Failed to parse point value"))
            .collect();
        points.push(point);
    }

    points
}

pub fn load_reference_scores() -> HashMap<String, Vec<f64>> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("testdata");
    path.push("reference-outlier-scores.csv");

    let contents = fs::read_to_string(&path).expect("Failed to read reference scores");
    let mut map = HashMap::new();

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }
        let name = parts[0];
        let scores: Vec<f64> = parts[1..]
            .iter()
            .map(|x| x.parse::<f64>().expect("Failed to parse reference score"))
            .collect();
        map.insert(name.to_string(), scores);
    }

    map
}

pub fn assert_outlier_scores_approx<F: Float>(
    method: &str, actual: &[F], expected: &[f64], tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len(), "{} length mismatch", method);
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        let a64 = a.to_f64().expect("Failed to convert score");

        let should_accept = if a64.is_infinite() && b.is_infinite() {
            a64.signum() == b.signum()
        } else {
            (a64 - b).abs() <= tolerance
        };

        assert!(should_accept, "{} mismatch at {}: {} vs {}", method, i, a64, b);
    }
}

pub fn assert_outlier_auc_approx(method: &str, perf_auc: f64, truth_auc: f64, tolerance: f64) {
    assert!(
        (perf_auc - truth_auc).abs() <= tolerance,
        "{} AUC mismatch: {} expected={}",
        method,
        perf_auc,
        truth_auc
    );
}

pub fn label_from_reference(reference: &HashMap<String, Vec<f64>>) -> Vec<u8> {
    let labels_ref = reference.get("bylabel").expect("bylabel missing in the reference data.");
    labels_ref.iter().map(|v| v.to_u8().expect("bylabel is supposed to be binary")).collect()
}

/// Sort a slice of outlier scores so that higher scores appear first.
/// Entropy over a probability-like vector using base-2 logarithm (bits).
/// Input values need not be normalized; they are normalized inside.
pub fn perplexity_to_entropy(perplexity: f64) -> f64 {
    if perplexity <= 0.0 {
        return 0.0;
    }
    perplexity.ln()
}

pub fn perplexity_entropy(values: &[f64]) -> f64 {
    let sum: f64 = values.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let mut h = 0.0;
    for &val in values.iter() {
        if val <= 0.0 {
            continue;
        }
        let p = val / sum;
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}
