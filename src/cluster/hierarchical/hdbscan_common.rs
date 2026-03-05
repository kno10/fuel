use std::cmp::Ordering;

use crate::DataAccess;

use super::common::{Merge, MergeHistory};

#[derive(Debug, Clone, PartialEq)]
pub struct HdbscanHierarchy {
    pub merges: MergeHistory<f64>,
    pub core_distances: Vec<f64>,
}

impl HdbscanHierarchy {
    #[must_use]
    pub const fn new(merges: MergeHistory<f64>, core_distances: Vec<f64>) -> Self {
        Self {
            merges,
            core_distances,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WeightedEdge {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) weight: f64,
}

impl WeightedEdge {
    #[must_use]
    pub(crate) const fn new(a: usize, b: usize, weight: f64) -> Self {
        Self { a, b, weight }
    }
}

pub(crate) fn compute_core_distances<D: DataAccess>(data: &D, min_points: usize) -> Vec<f64> {
    assert!(min_points > 0, "min_points must be greater than 0");

    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    if min_points > n {
        return vec![f64::INFINITY; n];
    }

    let rank = min_points - 1;
    let mut scratch = vec![0.0; n];
    let mut core_distances = vec![f64::INFINITY; n];

    for i in 0..n {
        for (j, slot) in scratch.iter_mut().enumerate() {
            *slot = data.distance(i, j);
        }
        let (_, kth, _) = scratch
            .select_nth_unstable_by(rank, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        core_distances[i] = *kth;
    }

    core_distances
}

pub(crate) fn mutual_reachability_distance<D: DataAccess>(
    data: &D,
    core_distances: &[f64],
    a: usize,
    b: usize,
) -> f64 {
    core_distances[a]
        .max(core_distances[b])
        .max(data.distance(a, b))
}

#[inline]
pub(crate) fn mutual_reachability_distance_from_distance(
    core_distances: &[f64],
    a: usize,
    b: usize,
    distance: f64,
) -> f64 {
    core_distances[a].max(core_distances[b]).max(distance)
}

pub(crate) fn edges_to_merge_history(n: usize, edges: &mut [WeightedEdge]) -> MergeHistory<f64> {
    if n <= 1 {
        return Vec::new();
    }

    edges.sort_by(|left, right| {
        left.weight
            .partial_cmp(&right.weight)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.a.min(left.b).cmp(&right.a.min(right.b)))
            .then_with(|| left.a.max(left.b).cmp(&right.a.max(right.b)))
    });

    let mut parent: Vec<usize> = (0..(2 * n - 1)).collect();
    let mut size = vec![1usize; 2 * n - 1];
    let mut merges = Vec::with_capacity(n - 1);

    for edge in edges.iter() {
        let s = uf_find(&mut parent, edge.a);
        let t = uf_find(&mut parent, edge.b);
        if s == t {
            continue;
        }

        let ss = size[s];
        let st = size[t];
        let (idx1, idx2) = if s <= t { (s, t) } else { (t, s) };
        merges.push(Merge {
            idx1,
            idx2,
            distance: edge.weight,
            size: ss + st,
        });

        let new_id = n + merges.len() - 1;
        parent[s] = new_id;
        parent[t] = new_id;
        parent[new_id] = new_id;
        size[new_id] = ss + st;

        if merges.len() == n - 1 {
            break;
        }
    }

    assert_eq!(merges.len(), n - 1, "edge set did not connect all points");
    merges
}

#[inline]
fn uf_find(parent: &mut [usize], x: usize) -> usize {
    let mut p = x;
    while parent[p] != p {
        p = parent[p];
    }
    let mut i = x;
    while parent[i] != p {
        let next = parent[i];
        parent[i] = p;
        i = next;
    }
    p
}
