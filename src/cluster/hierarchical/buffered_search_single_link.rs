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

#[derive(Default)]
struct BufferedNeighbors {
    heap: BinaryHeap<Neighbor>,
    threshold: f64,
}

/// Buffered-Search Single-Link (BSSL) with VP-tree priority search.
///
/// `slack` controls how many extra candidates are explored beyond the current
/// lower-bound threshold before stopping each refill phase.
#[must_use]
pub fn buffered_search_single_link<T, DF>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, DF>,
    slack: usize,
) -> MergeHistory<f64>
where
    DF: DistanceFunction<T>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");

    let mut builder = ClusterBuilder::new(n);
    let mut primary = BinaryHeap::new();
    let mut buffers: Vec<BufferedNeighbors> =
        (0..n).map(|_| BufferedNeighbors::default()).collect();
    let mut seen = vec![false; n];

    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        initialize_buffer(
            tree,
            data,
            &mut builder,
            a,
            slack,
            &mut buffers[a],
            &mut seen,
        );
        if let Some(top) = buffers[a].heap.peek().copied() {
            primary.push(QueueEntry {
                dist: top.dist,
                point: a,
            });
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = primary.pop() else {
            break;
        };
        let a = top.point;
        let Some(best) = buffers[a].heap.peek().copied() else {
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
        buffers[a].heap.pop();
        while let Some(c) = buffers[a].heap.peek().copied() {
            if builder.find(c.index) == ca {
                buffers[a].heap.pop();
            } else {
                break;
            }
        }

        if buffers[a]
            .heap
            .peek()
            .is_none_or(|next| next.dist > buffers[a].threshold)
        {
            refill_buffer(
                tree,
                data,
                &mut builder,
                a,
                top.dist,
                slack,
                &mut buffers[a],
                &mut seen,
            );
        }

        if let Some(next) = buffers[a].heap.peek().copied() {
            primary.push(QueueEntry {
                dist: next.dist,
                point: a,
            });
        }
    }

    builder.into_history()
}

fn initialize_buffer<T, DF>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, DF>,
    builder: &mut ClusterBuilder,
    a: usize,
    slack: usize,
    buffer: &mut BufferedNeighbors,
    seen: &mut [bool],
) where
    DF: DistanceFunction<T>,
{
    buffer.heap.clear();
    buffer.threshold = f64::INFINITY;
    refill_impl(tree, data, builder, a, 0.0, slack, buffer, seen);
}

fn refill_buffer<T, DF>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, DF>,
    builder: &mut ClusterBuilder,
    a: usize,
    skip: f64,
    slack: usize,
    buffer: &mut BufferedNeighbors,
    seen: &mut [bool],
) where
    DF: DistanceFunction<T>,
{
    refill_impl(tree, data, builder, a, skip, slack, buffer, seen);
}

fn refill_impl<T, DF>(
    tree: &VPTree,
    data: &MatrixDataAccess<'_, T, DF>,
    builder: &mut ClusterBuilder,
    a: usize,
    skip: f64,
    slack: usize,
    buffer: &mut BufferedNeighbors,
    seen: &mut [bool],
) where
    DF: DistanceFunction<T>,
{
    seen.fill(false);
    for item in buffer.heap.iter() {
        seen[item.index] = true;
    }

    let mut searcher = tree.priority_searcher(IndexedQueryData {
        data,
        query_index: a,
    });
    searcher.reset_with_limits(f64::INFINITY, skip.max(0.0));

    let mut threshold = buffer.heap.peek().map_or(f64::INFINITY, |n| n.dist);
    let mut remaining = slack as isize;
    loop {
        if searcher.all_lower_bound() >= threshold {
            if remaining <= 0 {
                break;
            }
            remaining -= 1;
        }
        let Some(cand) = searcher.next() else {
            break;
        };
        let b = cand.index();
        if a == b || seen[b] || builder.same_set(a, b) {
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
        buffer.heap.push(Neighbor { dist: d, index: b });
        seen[b] = true;
        threshold = buffer.heap.peek().map_or(f64::INFINITY, |n| n.dist);
    }
    buffer.threshold = searcher.all_lower_bound();
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
    fn buffered_matches_slink_on_unique_1d_distances() {
        let points = vec![vec![0.0], vec![1.1], vec![3.7], vec![10.2], vec![20.5]];
        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(11);
        let tree = VPTree::new(&data, 3, &mut rng);

        let expected = slink(&condensed_abs_1d(&points), points.len());
        let got = buffered_search_single_link(&tree, &data, 2);
        assert_eq!(got, expected);
    }
}
