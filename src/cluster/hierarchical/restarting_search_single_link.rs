use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand::rngs::StdRng;

#[cfg(test)]
use crate::EuclideanDistance;
use crate::{DataAccess, DistanceFunction, MatrixDataAccess, VPTree};

use super::common::MergeHistory;
use super::search_single_link_common::ClusterBuilder;

#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f64,
    point: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.dist == other.dist
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.point.cmp(&self.point))
    }
}

/// Restarting-Search Single-Link (RSSL) with VP-tree priority search.
///
/// This keeps one active nearest-neighbor candidate per point and refreshes it
/// by restarting search with an increased skip distance after each use.
#[must_use]
pub fn restarting_search_single_link<T, DF>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, DF>,
) -> MergeHistory<f64>
where
    DF: DistanceFunction<T>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut heap = BinaryHeap::new();
    let mut best_dist = vec![f64::INFINITY; n];
    let mut nn: Vec<Option<usize>> = vec![None; n];

    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        if let Some((d, b)) = find_next_neighbor(tree, data, &mut builder, a, 0.0) {
            best_dist[a] = d;
            nn[a] = Some(b);
            heap.push(Candidate { dist: d, point: a });
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = heap.pop() else {
            break;
        };
        let a = top.point;
        if nn[a].is_none() || top.dist.to_bits() != best_dist[a].to_bits() {
            continue;
        }
        let b = nn[a].expect("neighbor must be set");
        if !builder.same_set(a, b) {
            let _ = builder.merge_points(a, b, top.dist);
            if builder.merge_count() == n - 1 {
                break;
            }
        }

        let skip = top.dist;
        if let Some((d, nb)) = find_next_neighbor(tree, data, &mut builder, a, skip) {
            best_dist[a] = d;
            nn[a] = Some(nb);
            heap.push(Candidate { dist: d, point: a });
        } else {
            best_dist[a] = f64::INFINITY;
            nn[a] = None;
        }
    }

    builder.into_history()
}

fn find_next_neighbor<T, DF>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, DF>,
    builder: &mut ClusterBuilder,
    a: usize,
    skip: f64,
) -> Option<(f64, usize)>
where
    DF: DistanceFunction<T>,
{
    let mut searcher = tree.priority_searcher(super::search_single_link_common::IndexedQueryData {
        data,
        query_index: a,
    });
    searcher.reset_with_limits(f64::INFINITY, skip.max(0.0));

    let mut best_dist = f64::INFINITY;
    let mut best = None;
    while let Some(cand) = searcher.next() {
        let b = cand.index();
        if a == b || builder.same_set(a, b) {
            continue;
        }
        let d = cand.distance();
        if d == 0.0 {
            let _ = builder.merge_points(a, b, 0.0);
            continue;
        }
        if d < skip {
            continue;
        }
        if d < best_dist {
            best_dist = d;
            best = Some(b);
            searcher.decrease_cutoff(best_dist);
        }
    }
    best.map(|b| (best_dist, b))
}

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::slink;

    use super::*;

    fn condensed_abs_1d(points: &[Vec<f64>]) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 1..points.len() {
            for j in 0..i {
                out.push((points[i][0] - points[j][0]).abs());
            }
        }
        out
    }

    #[test]
    fn restarting_matches_slink_on_unique_1d_distances() {
        let points = vec![vec![0.0], vec![1.1], vec![3.7], vec![10.2], vec![20.5]];
        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(7);
        let tree = VPTree::new(&data, 3, &mut rng);

        let expected = slink(&condensed_abs_1d(&points), points.len());
        let got = restarting_search_single_link(&tree, &data);
        assert_eq!(got, expected);
    }
}
