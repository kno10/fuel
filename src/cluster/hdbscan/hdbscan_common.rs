use std::cmp::Ordering;

use crate::api::{NodePoints, SearchFilter};
use crate::cluster::hierarchical::common::{Merge, MergeHistory, UnionFind};
use crate::{DistanceData, Float, IndexQuery, KnnSearch};

#[derive(Debug, Clone, PartialEq)]
pub struct HdbscanHierarchy<F: Float> {
    pub merges: MergeHistory<F>,
    pub core_distances: Vec<F>,
}

impl<F: Float> HdbscanHierarchy<F> {
    #[must_use]
    pub const fn new(merges: MergeHistory<F>, core_distances: Vec<F>) -> Self {
        Self { merges, core_distances }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WeightedEdge<F: Float> {
    pub(crate) a: usize,
    pub(crate) b: usize,
    pub(crate) weight: F,
}

impl<F: Float> WeightedEdge<F> {
    #[must_use]
    pub(crate) const fn new(a: usize, b: usize, weight: F) -> Self { Self { a, b, weight } }
}

pub(crate) fn compute_core_distances<D: DistanceData<F>, F: Float>(
    data: &D, min_points: usize,
) -> Vec<F> {
    // fallback brute-force version used by non-tree algorithms (prim, slink)
    assert!(min_points > 0, "min_points must be greater than 0");

    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    if min_points > n {
        return vec![F::infinity(); n];
    }

    let rank = min_points - 1;
    let mut scratch = vec![F::zero(); n];
    let mut core_distances = vec![F::infinity(); n];

    for (i, cd) in core_distances.iter_mut().enumerate().take(n) {
        for (j, slot) in scratch.iter_mut().enumerate() {
            *slot = data.distance(i, j);
        }
        let (_, kth, _) = scratch
            .select_nth_unstable_by(rank, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        *cd = *kth;
    }

    core_distances
}

/// Compute core distances using a VP-tree for neighbourhood search.  This
/// avoids the quadratic cost of the brute-force routine above by asking the
/// tree for the `min_points` nearest neighbours of each point.  The tree
/// must have been built on the same dataset as `data`.
#[must_use]
pub fn compute_core_distances_tree<'a, S, D, F>(tree: &S, data: &'a D, min_points: usize) -> Vec<F>
where
    D: DistanceData<F> + ?Sized + 'a,
    F: Float,
    S: KnnSearch<F, D::Query<'a>>,
{
    assert!(min_points > 0, "min_points must be greater than 0");

    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    if min_points > n {
        return vec![F::infinity(); n];
    }

    let k = min_points; // search_knn returns self as first neighbour
    let mut core = vec![F::infinity(); n];
    let mut query = data.query();
    for (i, slot) in core.iter_mut().enumerate().take(n) {
        query.set_index(i);
        let neighbors = tree.search_knn(&query, k);
        if neighbors.len() == k {
            *slot = neighbors[k - 1].distance;
        }
    }

    core
}

#[inline]
pub(crate) fn mutual_reachability_distance<F: Float>(
    core_distances: &[F], a: usize, b: usize, distance: F,
) -> F {
    core_distances[a].max(core_distances[b]).max(distance)
}

pub(crate) struct SameComponentFilter<'a> {
    pub(crate) uf: &'a mut UnionFind,
    pub(crate) query_index: usize,
    pub(crate) node_cluster: &'a mut [u32],
}

const WITNESS_BIT: u32 = 1 << 31;

impl<'a> SearchFilter for SameComponentFilter<'a> {
    fn skip_node(&mut self, points: NodePoints<'_>) -> bool {
        let query_component = self.uf.find(self.query_index);
        let vp = points.first_index();
        let cached = self.node_cluster[vp];

        // Fast path: cached witness for non-uniformity
        if cached != u32::MAX && (cached & WITNESS_BIT) != 0 {
            let witness = (cached & !WITNESS_BIT) as usize;
            if self.uf.find(vp) != self.uf.find(witness) {
                return false;
            }
            // Witness merged, fall through to re-scan
        } else if cached != u32::MAX {
            // Fast path: cached uniform cluster
            return self.uf.find(cached as usize) == query_component;
        }

        let mut component = u32::MAX;
        for i in points.indices() {
            let c = self.uf.find(i) as u32;
            if component == u32::MAX {
                component = c;
            } else if c != component {
                self.node_cluster[vp] = WITNESS_BIT | (i as u32);
                return false;
            }
        }
        if component != u32::MAX {
            self.node_cluster[vp] = component;
        }
        component != u32::MAX && (component as usize) == query_component
    }

    fn skip_point(&mut self, index: usize) -> bool {
        index == self.query_index || self.uf.find(index) == self.uf.find(self.query_index)
    }
}

pub(crate) fn edges_to_merge_history<F: Float>(
    n: usize, edges: &mut [WeightedEdge<F>],
) -> MergeHistory<F> {
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
    let mut merges = Vec::<Merge<F>>::with_capacity(n - 1);

    for edge in edges.iter() {
        let s = uf_find(&mut parent, edge.a);
        let t = uf_find(&mut parent, edge.b);
        if s == t {
            continue;
        }

        let ss = size[s];
        let st = size[t];
        let (idx1, idx2) = if s <= t { (s, t) } else { (t, s) };
        merges.push(Merge { idx1, idx2, distance: edge.weight, size: ss + st });

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

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::EuclideanDistance;
    use crate::vptree::VPTree;

    #[test]
    fn core_distances_tree_matches_bruteforce() {
        let points: Vec<Vec<f64>> =
            vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]];
        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(123);
        let tree = VPTree::<f64>::new(&data, 2, &mut rng);

        for min_points in 1..=5 {
            let brute: Vec<f64> = compute_core_distances(&data, min_points);
            let via_tree = compute_core_distances_tree(&tree, &data, min_points);
            assert_eq!(brute, via_tree);
        }
    }
}
