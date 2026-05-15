//! Shared helper utilities for hierarchical clustering algorithms.
//!
//! This module contains common utilities used by AGNES, Anderberg-family
//! implementations, set-based clustering, and other hierarchical methods.
//! It includes helpers for condensed distance indexing, nearest-neighbor cache
//! maintenance, and common set-clustering support.

use num_traits::NumCast;

use crate::cluster::hierarchical::{SetLinkage, idsize};
use crate::{CandidateHeap, DistPair, DistanceData, Float};

/// Compute the index in the condensed array for pair `(i, j)`, assuming
/// `i > j`.
///
/// This mirrors the inline helper formerly defined inside `agnes.rs`; we
/// pull it out here so that the core algorithm stays focused on the
/// clustering logic.
#[inline(always)]
#[must_use]
pub(crate) const fn triangle_index(i: usize, j: usize) -> usize {
    debug_assert!(i != j, "no diagonal entries in condensed matrix");
    let (a, b) = if i >= j { (i, j) } else { (j, i) };
    // Note: a > b, hence a > 0
    (a * (a - 1)) / 2 + b
}

#[inline]
pub(crate) fn condensed_get<F: Copy>(mat: &[F], a: usize, b: usize) -> F {
    // triangle_index handles ordering
    mat[triangle_index(a, b)]
}

#[inline]
pub(crate) fn condensed_set<F>(mat: &mut [F], a: usize, b: usize, value: F) {
    mat[triangle_index(a, b)] = value;
}

/// Small buffer used by several search algorithms.
#[derive(Default)]
pub(crate) struct BufferedNeighbors<F: Float> {
    pub(crate) heap: CandidateHeap<F>,
    pub(crate) threshold: F,
}

impl<F: Float> BufferedNeighbors<F>
where
    F: Float,
{
    pub(crate) fn new() -> Self { Self { heap: CandidateHeap::new(), threshold: F::infinity() } }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool { self.heap.is_empty() }

    #[inline]
    pub(crate) fn push(&mut self, n: DistPair<F>) { self.heap.push(n) }

    #[inline]
    pub(crate) fn pop(&mut self) -> Option<DistPair<F>> { self.heap.pop() }

    #[inline]
    pub(crate) fn peek(&self) -> Option<DistPair<F>> { self.heap.peek() }

    /// Reset the buffer to its initial empty state.
    #[inline]
    pub(crate) fn reset(&mut self) -> &mut Self {
        self.heap.clear();
        self.threshold = F::infinity();
        self
    }
}

#[inline]
pub(crate) fn find_active(start: usize, end: usize, clustermap: &[idsize]) -> Option<usize> {
    (start..end).find(|&i| clustermap[i] != idsize::MAX)
}

#[inline]
pub(crate) fn shrink_active_end(clustermap: &[idsize], end: &mut usize) {
    while *end > 0 && clustermap[*end - 1] == idsize::MAX {
        *end -= 1;
    }
}

// ----------------------------------------------------------------------
// Nearest-neighbour cache utilities

pub(crate) fn initialize_set_clusters<D, L, F, S>(
    data: &D,
) -> Result<(Vec<Vec<idsize>>, Vec<S>, Vec<F>, Vec<idsize>), String>
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let n = data.len();
    let members: Vec<Vec<idsize>> = (0..n).map(|i| vec![i as idsize]).collect();
    let summaries: Vec<S> = members.iter().map(|m| L::summarize(data, m)).collect();

    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    for x in 1..n {
        crate::poll_interrupted()?;
        for y in 0..x {
            // TODO: retain/cache the pair-wise summaries?
            let (d, _) =
                L::cluster_distance(data, &summaries[x], &summaries[y], &members[x], &members[y]);
            distances.push(d);
        }
    }

    Ok((members, summaries, distances, (0..(n as idsize)).collect()))
}

/// Update the condensed-distance matrix for the pair `(x,y)` according to a
/// `SetLinkage` implementation.  This was previously duplicated in several of
/// the `set_*` algorithms; factoring it simplifies those modules.
#[inline]
pub(crate) fn update_set_entry<D, L, F, S>(
    data: &D, distances: &mut [F], members: &[Vec<idsize>], summaries: &[S], x: usize, y: usize,
) where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let (d, _) = L::cluster_distance(data, &summaries[x], &summaries[y], &members[x], &members[y]);
    distances[triangle_index(x, y)] = d;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_update_cache<F>(
    distances: &[F], clustermap: &[idsize], best: &mut [(F, idsize)], x: usize, y: usize, j: usize,
    d: F,
) -> bool
where
    F: Float,
{
    if y < j {
        let old = best[j];
        if d <= old.0 {
            best[j] = (d, y as idsize);
            return true;
        }
    }

    if best[j].1 == x as idsize || best[j].1 == y as idsize {
        let old = best[j];
        set_find_best::<F>(distances, clustermap, best, j);
        return best[j] != old;
    }

    false
}

pub(crate) fn set_find_best<F>(
    distances: &[F], clustermap: &[idsize], best: &mut [(F, idsize)], j: usize,
) where
    F: Float,
{
    let mut best_pair = (F::infinity(), idsize::MAX);

    for i in 0..j {
        if clustermap[i] == idsize::MAX {
            continue;
        }
        let raw = distances[triangle_index(j, i)];
        if raw < best_pair.0 {
            best_pair = (raw, i as idsize);
        }
    }

    best[j] = best_pair;
}

pub(crate) fn set_find_best_active_pair<F>(
    distances: &[F], clustermap: &[idsize], end: usize,
) -> (usize, usize, F)
where
    F: Float,
{
    let mut best = (F::infinity(), usize::MAX, usize::MAX);

    for x in 1..end {
        if clustermap[x] == idsize::MAX {
            continue;
        }
        for y in 0..x {
            if clustermap[y] == idsize::MAX {
                continue;
            }
            let raw = distances[triangle_index(x, y)];
            if raw < best.0 {
                best = (raw, x, y);
            }
        }
    }

    assert!(best.1 != usize::MAX && best.2 != usize::MAX, "no merge candidate found");
    (best.1.max(best.2), best.1.min(best.2), best.0)
}

/// Union-find used by multiple clustering routines.
///
/// This implementation uses path compression during `find` and
/// union-by-size during `union` to keep tree height low and improve
/// amortized performance.
#[derive(Debug, Clone)]
pub(crate) struct UnionFind<U = usize> {
    pub parent: Vec<U>,
    pub size: Vec<usize>,
}

impl<U> UnionFind<U>
where
    U: Copy + NumCast + PartialEq + Eq,
{
    pub(crate) fn new(n: usize) -> Self {
        Self { parent: (0..n).map(|i| NumCast::from(i).unwrap()).collect(), size: vec![1; n] }
    }

    pub(crate) fn find(&mut self, x: U) -> U {
        let mut root = x;
        while {
            let root_idx: usize = num_traits::cast(root).unwrap();
            self.parent[root_idx] != root
        } {
            let root_idx: usize = num_traits::cast(root).unwrap();
            root = self.parent[root_idx];
        }
        let mut node = x;
        while {
            let node_idx: usize = num_traits::cast(node).unwrap();
            self.parent[node_idx] != root
        } {
            let node_idx: usize = num_traits::cast(node).unwrap();
            let next = self.parent[node_idx];
            self.parent[node_idx] = root;
            node = next;
        }
        root
    }

    pub(crate) fn union(&mut self, a: U, b: U) -> (bool, U) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return (false, ra);
        }
        let mut ra_idx: usize = num_traits::cast(ra).unwrap();
        let mut rb_idx: usize = num_traits::cast(rb).unwrap();
        if self.size[ra_idx] < self.size[rb_idx] {
            std::mem::swap(&mut ra, &mut rb);
            std::mem::swap(&mut ra_idx, &mut rb_idx);
        }
        self.parent[rb_idx] = ra;
        self.size[ra_idx] += self.size[rb_idx];
        (true, ra)
    }
}
