use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(test)]
use rand::SeedableRng;
#[cfg(test)]
use rand::rngs::StdRng;

#[cfg(test)]
use crate::EuclideanDistance;
use crate::{DataAccess, DistanceFunction, MatrixDataAccess, PrioritySearcher, VPTree};

use super::common::MergeHistory;
use super::search_single_link_common::{ClusterBuilder, IndexedQueryData};

#[derive(Debug, Clone, Copy)]
struct Neighbor {
    dist: f64,
    index: usize,
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.dist == other.dist
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

#[derive(Debug, Clone, Copy)]
struct QueueEntry {
    dist: f64,
    point: usize,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.dist == other.dist
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.point.cmp(&self.point))
    }
}

/// Heap-of-Searchers Single-Link (HSSL) with VP-tree priority search.
#[must_use]
pub fn heap_of_searchers_single_link<'t, 'm, 'd, T, DF>(
    tree: &'t VPTree,
    data: &'m MatrixDataAccess<'d, T, DF>,
) -> MergeHistory<f64>
where
    DF: DistanceFunction<T>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut primary = BinaryHeap::new();
    let mut neighbor_heaps: Vec<BinaryHeap<Neighbor>> = (0..n).map(|_| BinaryHeap::new()).collect();
    let mut searchers: Vec<Option<PrioritySearcher<'t, IndexedQueryData<'m, 'd, T, DF>, f64>>> =
        (0..n).map(|_| None).collect();

    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        let mut searcher = tree.priority_searcher(IndexedQueryData {
            data,
            query_index: a,
        });
        initialize_neighbors(&mut searcher, &mut builder, a, &mut neighbor_heaps[a]);
        if let Some(top) = neighbor_heaps[a].peek().copied() {
            primary.push(QueueEntry {
                dist: top.dist,
                point: a,
            });
            searchers[a] = Some(searcher);
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = primary.pop() else {
            break;
        };
        let a = top.point;
        let Some(best) = neighbor_heaps[a].peek().copied() else {
            continue;
        };
        if best.dist.to_bits() != top.dist.to_bits() {
            continue;
        }

        let b = best.index;
        if !builder.same_set(a, b) {
            let _ = builder.merge_points(a, b, top.dist);
            if builder.merge_count() == n - 1 {
                break;
            }
        }

        let ca = builder.find(a);
        neighbor_heaps[a].pop();
        while let Some(next) = neighbor_heaps[a].peek().copied() {
            if builder.find(next.index) == ca {
                neighbor_heaps[a].pop();
            } else {
                break;
            }
        }

        let needs_refill = neighbor_heaps[a].peek().is_none_or(|next| {
            next.dist
                > searchers[a]
                    .as_ref()
                    .expect("searcher must exist when heap exists")
                    .all_lower_bound()
        });
        if needs_refill {
            if let Some(searcher) = searchers[a].as_mut() {
                refill_neighbors(searcher, &mut builder, a, &mut neighbor_heaps[a]);
            }
        }

        if let Some(next) = neighbor_heaps[a].peek().copied() {
            primary.push(QueueEntry {
                dist: next.dist,
                point: a,
            });
        } else {
            searchers[a] = None;
        }
    }

    builder.into_history()
}

fn initialize_neighbors<T, DF>(
    searcher: &mut PrioritySearcher<'_, IndexedQueryData<'_, '_, T, DF>, f64>,
    builder: &mut ClusterBuilder,
    a: usize,
    heap: &mut BinaryHeap<Neighbor>,
) where
    DF: DistanceFunction<T>,
{
    let mut threshold = f64::INFINITY;
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next() else {
            break;
        };
        let b = cand.index();
        if a == b {
            continue;
        }
        let d = cand.distance();
        if d == 0.0 {
            let _ = builder.merge_points(a, b, 0.0);
            continue;
        }
        heap.push(Neighbor { dist: d, index: b });
        threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    }
}

fn refill_neighbors<T, DF>(
    searcher: &mut PrioritySearcher<'_, IndexedQueryData<'_, '_, T, DF>, f64>,
    builder: &mut ClusterBuilder,
    a: usize,
    heap: &mut BinaryHeap<Neighbor>,
) where
    DF: DistanceFunction<T>,
{
    let mut threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    while searcher.all_lower_bound() < threshold {
        let Some(cand) = searcher.next() else {
            break;
        };
        let b = cand.index();
        if a == b || builder.same_set(a, b) {
            continue;
        }
        heap.push(Neighbor {
            dist: cand.distance(),
            index: b,
        });
        threshold = heap.peek().map_or(f64::INFINITY, |n| n.dist);
    }
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
    fn hssl_matches_slink_on_unique_1d_distances() {
        let points = vec![vec![0.0], vec![1.1], vec![3.7], vec![10.2], vec![20.5]];
        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(17);
        let tree = VPTree::new(&data, 3, &mut rng);

        let expected = slink(&condensed_abs_1d(&points), points.len());
        let got = heap_of_searchers_single_link(&tree, &data);
        assert_eq!(got, expected);
    }
}
