use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::outlier::common::{OutlierResult, make_outlier_result};
use crate::{DistanceData, Float, VectorData};

const EULER_MASCHERONI: f64 = 0.577_215_664_901_532_9;

fn c_factor(n: f64) -> f64 {
    if n <= 1.0 { 0.0 } else { 2.0 * ((n - 1.0).ln() + EULER_MASCHERONI) - (2.0 * (n - 1.0) / n) }
}

enum IsolationTreeNode {
    External(usize),
    Internal {
        axis: usize,
        split: f64,
        left: Box<IsolationTreeNode>,
        right: Box<IsolationTreeNode>,
    },
}

fn build_isolation_tree<F: Float, D>(
    data: &D, indices: &[usize], depth: usize, max_height: usize, rng: &mut StdRng,
) -> IsolationTreeNode
where
    D: DistanceData<F> + VectorData<F>,
{
    let size = indices.len();
    if depth >= max_height || size <= 1 {
        return IsolationTreeNode::External(size);
    }

    let dim = data.dims();

    let mut minv = vec![f64::INFINITY; dim];
    let mut maxv = vec![f64::NEG_INFINITY; dim];
    for &idx in indices.iter() {
        let point = data.point(idx);
        for d in 0..dim {
            let v = point[d].to_f64().unwrap_or(0.0);
            if v < minv[d] {
                minv[d] = v;
            }
            if v > maxv[d] {
                maxv[d] = v;
            }
        }
    }

    if (0..dim).all(|d| maxv[d] <= minv[d]) {
        return IsolationTreeNode::External(size);
    }

    let mut active = Vec::new();
    for d in 0..dim {
        if maxv[d] > minv[d] {
            active.push(d);
        }
    }

    if active.is_empty() {
        return IsolationTreeNode::External(size);
    }

    let axis = active[rng.random_range(0..active.len())];
    let split = minv[axis] + (maxv[axis] - minv[axis]) * rng.random_range(0.0..1.0);

    let mut left = Vec::new();
    let mut right = Vec::new();
    for &idx in indices.iter() {
        let v = data.point(idx)[axis].to_f64().unwrap_or(0.0);
        if v <= split {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }

    if left.is_empty() || right.is_empty() {
        return IsolationTreeNode::External(size);
    }

    IsolationTreeNode::Internal {
        axis,
        split,
        left: Box::new(build_isolation_tree(data, &left, depth + 1, max_height, rng)),
        right: Box::new(build_isolation_tree(data, &right, depth + 1, max_height, rng)),
    }
}

fn isolation_path_length<F: Float, D>(
    node: &IsolationTreeNode, data: &D, query_point: usize, depth: f64,
) -> f64
where
    D: DistanceData<F> + VectorData<F>,
{
    match node {
        IsolationTreeNode::External(size) => {
            let depth = depth + 1.0;
            if *size <= 1 { depth } else { depth + c_factor(*size as f64) }
        }
        IsolationTreeNode::Internal { axis, split, left, right } => {
            let query_coord = data.point(query_point)[*axis].to_f64().unwrap_or(0.0);
            if query_coord <= *split {
                isolation_path_length(left, data, query_point, depth + 1.0)
            } else {
                isolation_path_length(right, data, query_point, depth + 1.0)
            }
        }
    }
}

pub fn isolation_forest<'a, D, F>(
    data: &'a D, num_trees: usize, subsample_size: usize, seed: u64,
) -> OutlierResult<F>
where
    F: Float,
    D: DistanceData<F> + VectorData<F> + Sync + 'a,
{
    let n = data.len();
    if n == 0 || subsample_size < 2 {
        return make_outlier_result(
            Vec::new(),
            "IsolationForest",
            false,
            F::zero(),
            F::zero(),
            F::one(),
        );
    }

    let subsample_size = subsample_size.min(n);
    let mut rng = StdRng::seed_from_u64(seed);
    let c = c_factor(subsample_size as f64);
    let mut path_sum = vec![0.0; n];

    let idxs: Vec<usize> = (0..n).collect();
    let max_height = (subsample_size as f64).log2().ceil() as usize;

    for _ in 0..num_trees {
        let mut sample = idxs.clone();
        sample.shuffle(&mut rng);
        let sample = &sample[0..subsample_size];

        let tree = build_isolation_tree(data, sample, 0, max_height, &mut rng);
        for (i, path) in path_sum.iter_mut().enumerate().take(n) {
            let depth = isolation_path_length(&tree, data, i, 0.0);
            *path += depth;
        }
    }

    let scores: Vec<F> = (0..n)
        .map(|i| {
            let avg_path = path_sum[i] / (num_trees as f64);
            let score = 2f64.powf(-avg_path / c);
            F::from_f64(score).unwrap_or(F::zero())
        })
        .collect();

    make_outlier_result(scores, "IsolationForest", false, F::zero(), F::zero(), F::infinity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::Euclidean;
    use crate::evaluation::outlier::receiver_operating_curve::auroc;
    use crate::outlier::common::*;
    use crate::{Data, TableWithDistance};

    #[test]
    fn isolation_forest_test() {
        let points = vec![vec![0.0], vec![0.1], vec![0.2], vec![10.0]];
        let data = TableWithDistance::with_distance(&points, Euclidean);
        let results = isolation_forest(&data, 20, 3, 42);
        let (best_index, _) = results
            .scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert_eq!(best_index, 3);
    }

    #[test]
    fn isolation_forest_full_matches_reference_outlier_score() {
        let points = load_gaussian4d_points();
        let data = TableWithDistance::with_distance(&points, Euclidean);

        let results = isolation_forest(&data, 100, data.len(), 42);
        let reference = load_reference_scores();
        let expected = reference.get("IForest-full").expect("No reference for IForest-full");
        let labels: Vec<u8> = label_from_reference(&reference);

        assert_outlier_auc_approx(
            "IForest-full",
            auc(&results.scores, &labels),
            auc(expected, &labels),
            1e-2,
        );
        assert_outlier_scores_approx("IForest-full", &results.scores, expected, 1e-1);
    }
}
