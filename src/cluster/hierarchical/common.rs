#![allow(clippy::type_complexity)]

//! Helper utilities used by the AGNES implementation.
//!
//! The algorithm relies on a small helper for building the merge history
//! while tracking cluster sizes, along with a convenience function for
//! computing indices into the condensed distance matrix.

use crate::cluster::hierarchical::SetLinkage;
use crate::cluster::hierarchical::linkage::Linkage;
use crate::{CandidateHeap, DistPair, DistanceData, Float};

/// Compute the index in the condensed array for pair `(i, j)`, assuming
/// `i > j`.
///
/// This mirrors the inline helper formerly defined inside `agnes.rs`; we
/// pull it out here so that the core algorithm stays focused on the
/// clustering logic.
#[inline]
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
pub(crate) fn find_active(start: usize, end: usize, clustermap: &[Option<usize>]) -> Option<usize> {
    (start..end).find(|&i| clustermap[i].is_some())
}

#[inline]
pub(crate) fn shrink_active_end(clustermap: &[Option<usize>], end: &mut usize) {
    while *end > 0 && clustermap[*end - 1].is_none() {
        *end -= 1;
    }
}

// ----------------------------------------------------------------------
// Nearest-neighbour cache utilities
//
// Historically these helpers lived in `nn_cache.rs`; the algorithms that
// make use of them are now numerous enough that it makes more sense to keep
// them alongside the other small helpers in `common.rs`.  The implementations
// are generic over `Float` so they can be reused by both `f32` and `f64`
// variants when necessary.

/// Initialise nearest-neighbour information for every active cluster index.
///
/// `bestd` and `besti` are parallel slices containing the current best
/// distance and index for each entry; entries corresponding to inactive
/// clusters should already have been set to infinity / `usize::MAX`.  This
/// simply runs `find_best` for each active position.
#[inline]
pub(crate) fn initialize_nn_cache<F: Float>(
    distances: &[F], clustermap: &[Option<usize>], bestd: &mut [F], besti: &mut [usize],
) {
    let size = bestd.len();
    for x in 1..size {
        if clustermap[x].is_none() {
            continue;
        }
        find_best(distances, clustermap, bestd, besti, x);
    }
}

/// Scan the nearest‑neighbour arrays to locate the pair of active clusters
/// with the smallest recorded distance.
///
/// The return tuple is `(distance, x, y)` with `x > y` guaranteed.
/// Appropriate guards are applied to skip inactive indices or those with no
/// known neighbour.
#[inline]
pub(crate) fn find_merge_scan<F: Float>(
    bestd: &[F], besti: &[usize], clustermap: &[Option<usize>], end: usize,
) -> (F, usize, usize) {
    let mut mindist = F::infinity();
    let mut x = usize::MAX;
    let mut y = usize::MAX;

    for cx in 1..end {
        if clustermap[cx].is_none() || besti[cx] == usize::MAX {
            continue;
        }
        let d = bestd[cx];
        if d <= mindist {
            mindist = d;
            x = cx;
            y = besti[cx];
        }
    }

    assert!(x != usize::MAX && y != usize::MAX, "no merge candidate found");
    if y < x { (mindist, x, y) } else { (mindist, y, x) }
}

/// Attempt to update the nearest‑neighbour cache entry for `j` after a new
/// distance `d` has been computed for the pair `(y, j)` during a merge
/// operation.  Returns `true` if the cache entry was modified.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn update_cache<F: Float>(
    distances: &[F], clustermap: &[Option<usize>], bestd: &mut [F], besti: &mut [usize], x: usize,
    y: usize, j: usize, d: F,
) -> bool {
    if y < j && d <= bestd[j] {
        bestd[j] = d;
        besti[j] = y;
        return true;
    }

    if besti[j] == x || besti[j] == y {
        let oldd = bestd[j];
        let oldi = besti[j];
        find_best(distances, clustermap, bestd, besti, j);
        return besti[j] != oldi || bestd[j] != oldd;
    }

    false
}

/// Shared matrix update logic for Andersberg-style nearest-neighbor caches.
///
/// `on_update` is invoked whenever the cache entry for `j` actually changed so
/// callers can react (e.g. by pushing new candidates into a heap).
#[allow(clippy::too_many_arguments)]
pub(crate) fn update_matrix_and_cache_with_hook<F, L, OnUpdate>(
    mat: &mut [F], clustermap: &[Option<usize>], bestd: &mut [F], besti: &mut [usize],
    builder: &Builder<F>, linkage: L, mindist: F, x: usize, y: usize, size_x: usize, size_y: usize,
    end: usize, mut on_update: OnUpdate,
) where
    F: Float,
    L: Linkage<F> + Copy,
    OnUpdate: FnMut(&[F], &[usize], usize),
{
    for j in 0..end {
        if j == x || j == y || clustermap[j].is_none() {
            continue;
        }

        let d_xj = condensed_get(mat, x, j);
        let d_yj = condensed_get(mat, y, j);
        let size_j = builder.get_size(clustermap[j].expect("j must be active"));
        let d = linkage.combine(size_x, d_xj, size_y, d_yj, size_j, mindist);
        condensed_set(mat, y, j, d);

        if update_cache(mat, clustermap, bestd, besti, x, y, j, d) {
            on_update(bestd, besti, j);
        }
    }
}

/// Abstract builder interface used by the Anderberg helper.
pub(crate) trait AgglomerativeBuilder<F: Float> {
    type Output;

    fn new(n: usize) -> Self;
    fn get_size(&self, cid: usize) -> usize;
    fn add(&mut self, a: usize, dist: F, b: usize, prototype: Option<usize>) -> usize;
    fn into_merges(self) -> Self::Output;
}

pub(crate) fn run_anderberg_nn_cache<F, B, Update, Recompute, Prepare>(
    mut distances: Vec<F>, n: usize, mut update: Update, mut recompute: Recompute,
    mut prepare: Prepare, sort_pairs: bool, prototypes: &mut [Option<usize>],
) -> B::Output
where
    F: Float,
    B: AgglomerativeBuilder<F>,
    Update: FnMut(
        &mut Vec<F>,
        &mut Vec<Option<usize>>,
        &mut B,
        &mut [F],
        &mut [usize],
        usize,
        usize,
        F,
        usize,
        usize,
        usize,
        usize,
        &mut [Option<usize>],
        Option<usize>,
    ),
    Recompute: FnMut(usize, usize, &Vec<Option<usize>>, &mut Vec<F>, &mut [F], &mut [usize]),
    Prepare: FnMut(F, usize, usize, usize, &[Option<usize>]) -> (F, Option<usize>),
{
    let mut builder = B::new(n);
    let mut clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();
    let mut end = n;

    let mut bestd = vec![F::infinity(); n];
    let mut besti = vec![usize::MAX; n];
    initialize_nn_cache(&distances, &clustermap, &mut bestd, &mut besti);

    for _ in 1..n {
        let (mindist, x, y) = find_merge_scan(&bestd, &besti, &clustermap, end);
        let cid_x = clustermap[x].expect("x must be active");
        let cid_y = clustermap[y].expect("y must be active");
        let size_x = builder.get_size(cid_x);
        let size_y = builder.get_size(cid_y);

        let offset = triangle_index(x, y);
        let (record_dist, prototype) = prepare(mindist, x, y, offset, prototypes);
        let (a, b) = if sort_pairs {
            if cid_y <= cid_x { (cid_y, cid_x) } else { (cid_x, cid_y) }
        } else {
            (cid_x, cid_y)
        };
        let new_id = builder.add(a, record_dist, b, prototype);

        clustermap[y] = Some(new_id);
        clustermap[x] = None;
        besti[x] = usize::MAX;
        bestd[x] = F::infinity();

        update(
            &mut distances,
            &mut clustermap,
            &mut builder,
            &mut bestd,
            &mut besti,
            x,
            y,
            mindist,
            end,
            size_x,
            size_y,
            offset,
            prototypes,
            prototype,
        );

        recompute(y, x, &clustermap, &mut distances, &mut bestd, &mut besti);

        if x == end - 1 {
            shrink_active_end(&clustermap, &mut end);
        }
    }

    builder.into_merges()
}

/// Find the best neighbour for cluster index `j` among all earlier active
/// clusters.  This is the workhorse used by both `initialize_nn_cache` and
/// `update_cache`.
#[inline]
pub(crate) fn find_best<F: Float>(
    distances: &[F], clustermap: &[Option<usize>], bestd: &mut [F], besti: &mut [usize], j: usize,
) {
    let mut best_dist = F::infinity();
    let mut best_idx = usize::MAX;

    for i in 0..j {
        if clustermap[i].is_none() {
            continue;
        }
        let d = distances[triangle_index(j, i)];
        if d <= best_dist {
            best_dist = d;
            best_idx = i;
        }
    }

    bestd[j] = best_dist;
    besti[j] = best_idx;
}

pub(crate) fn initialize_set_clusters<D, L, F, S>(
    data: &D,
) -> (Vec<Vec<usize>>, Vec<Option<S>>, Vec<F>, Vec<Option<usize>>, Vec<Option<usize>>)
where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let n = data.size();
    let members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let summaries: Vec<Option<S>> = members.iter().map(|m| Some(L::summarize(data, m))).collect();

    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    let mut prototypes: Vec<Option<usize>> = Vec::with_capacity(n * (n - 1) / 2);
    for x in 1..n {
        for y in 0..x {
            let summary_x = summaries[x].as_ref().expect("summary missing for active cluster");
            let summary_y = summaries[y].as_ref().expect("summary missing for active cluster");
            let (d, proto) =
                L::cluster_distance(data, summary_x, summary_y, &members[x], &members[y]);
            distances.push(d);
            prototypes.push(proto);
        }
    }

    let clustermap: Vec<Option<usize>> = (0..n).map(Some).collect();

    (members, summaries, distances, prototypes, clustermap)
}

/// Update the condensed-distance matrix and prototype vector for the pair
/// `(x,y)` according to a `SetLinkage` implementation.  This was previously
/// duplicated in several of the `set_*` algorithms; factoring it simplifies
/// those modules.
#[inline]
pub(crate) fn update_set_entry<D, L, F, S>(
    data: &D, distances: &mut [F], prototypes: &mut [Option<usize>], members: &[Vec<usize>],
    summaries: &[Option<S>], x: usize, y: usize,
) where
    D: DistanceData<F>,
    F: Float,
    L: SetLinkage<D, F, S>,
{
    let summary_x = summaries[x].as_ref().expect("summary missing for active cluster");
    let summary_y = summaries[y].as_ref().expect("summary missing for active cluster");
    let (d, proto) = L::cluster_distance(data, summary_x, summary_y, &members[x], &members[y]);
    let offset = triangle_index(x, y);
    distances[offset] = d;
    prototypes[offset] = proto;
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_update_cache<D, F, L, S>(
    distances: &[F], clustermap: &[Option<usize>], summaries: &[Option<S>], bestd: &mut [F],
    besti: &mut [usize], x: usize, y: usize, j: usize, d: F,
) -> bool
where
    F: Float,
    D: DistanceData<F>,
    L: SetLinkage<D, F, S>,
{
    let summary_y = summaries[y].as_ref().expect("summary missing for active cluster");
    let summary_j = summaries[j].as_ref().expect("summary missing for active cluster");
    let adjusted = L::adjust_distance(d, summary_y, summary_j);
    if y < j && adjusted <= bestd[j] {
        bestd[j] = adjusted;
        besti[j] = y;
        return true;
    }

    if besti[j] == x || besti[j] == y {
        let oldd = bestd[j];
        let oldi = besti[j];
        set_find_best::<D, F, L, S>(distances, clustermap, summaries, bestd, besti, j);
        return besti[j] != oldi || bestd[j] != oldd;
    }

    false
}

pub(crate) fn set_find_best<D, F, L, S>(
    distances: &[F], clustermap: &[Option<usize>], summaries: &[Option<S>], bestd: &mut [F],
    besti: &mut [usize], j: usize,
) where
    F: Float,
    D: DistanceData<F>,
    L: SetLinkage<D, F, S>,
{
    let mut best_dist = F::infinity();
    let mut best_idx = usize::MAX;
    let summary_j = summaries[j].as_ref().expect("summary missing for active cluster");

    for i in 0..j {
        if clustermap[i].is_none() {
            continue;
        }
        let summary_i = summaries[i].as_ref().expect("summary missing for active cluster");
        let raw = distances[triangle_index(j, i)];
        let adjusted = L::adjust_distance(raw, summary_i, summary_j);
        if adjusted <= best_dist {
            best_dist = adjusted;
            best_idx = i;
        }
    }

    bestd[j] = best_dist;
    besti[j] = best_idx;
}

pub(crate) fn set_find_best_active_pair<D, F, L, S>(
    distances: &[F], clustermap: &[Option<usize>], summaries: &[Option<S>], end: usize,
) -> (usize, usize, F)
where
    F: Float,
    D: DistanceData<F>,
    L: SetLinkage<D, F, S>,
{
    let mut best_dist = F::infinity();
    let mut best_x = usize::MAX;
    let mut best_y = usize::MAX;

    for x in 1..end {
        if clustermap[x].is_none() {
            continue;
        }
        let summary_x = summaries[x].as_ref().expect("summary missing for active cluster");
        for y in 0..x {
            if clustermap[y].is_none() {
                continue;
            }
            let summary_y = summaries[y].as_ref().expect("summary missing for active cluster");
            let raw = distances[triangle_index(x, y)];
            let adjusted = L::adjust_distance(raw, summary_x, summary_y);
            if adjusted <= best_dist {
                best_dist = adjusted;
                best_x = x;
                best_y = y;
            }
        }
    }

    assert!(best_x != usize::MAX && best_y != usize::MAX, "no merge candidate found");

    (best_x.max(best_y), best_x.min(best_y), best_dist)
}

/// Simple union-find used by multiple clustering routines.
#[derive(Debug, Clone)]
pub(crate) struct UnionFind {
    pub parent: Vec<usize>,
    pub size: Vec<usize>,
}

impl UnionFind {
    pub(crate) fn new(n: usize) -> Self { Self { parent: (0..n).collect(), size: vec![1; n] } }

    pub(crate) fn find(&mut self, x: usize) -> usize {
        let mut root = x;
        while self.parent[root] != root {
            root = self.parent[root];
        }
        let mut node = x;
        while self.parent[node] != root {
            let next = self.parent[node];
            self.parent[node] = root;
            node = next;
        }
        root
    }

    pub(crate) fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
        true
    }
}

/// A single merge event in a hierarchical clustering history.
///
/// Each entry corresponds to the tuple `(i, j, d, s)` used by `SciPy`'s
/// `linkage` output: `i` and `j` are cluster identifiers (initial points
/// have ids `0..n-1`, merged clusters are numbered `n..`), `d` is the merge
/// distance, and `s` is the size of the new cluster.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Merge<F: Float> {
    pub idx1: usize,
    pub idx2: usize,
    pub distance: F,
    pub size: usize,
}

/// Convenience alias for the full merge history produced by an agglomerative
/// algorithm.  Typically this will contain `n-1` entries for `n` input points.
pub type MergeHistory<F> = Vec<Merge<F>>;

/// A merge event that also tracks a prototype (e.g. medoid) representative.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PrototypeMerge<F: Float> {
    pub idx1: usize,
    pub idx2: usize,
    pub distance: F,
    pub size: usize,
    pub prototype: Option<usize>,
}

/// Merge history containing prototype information per merge.
pub type PrototypeMergeHistory<F> = Vec<PrototypeMerge<F>>;

/// Builder for merge history and cluster sizes.
///
/// The original AGNES implementation kept this as a small internal struct.
/// Extracting it into its own module makes `agnes.rs` easier to read and
/// allows potential reuse by other clustering routines in the future.
pub(crate) struct Builder<F: Float> {
    n: usize,
    merges: MergeHistory<F>,
    sizes: Vec<usize>, // size for each cluster id, including merged ones
}

impl<F: Float> Builder<F> {
    pub(crate) fn new(n: usize) -> Self {
        // maximum of `2*n - 1` cluster ids may be referenced
        let mut sizes = Vec::with_capacity(2 * n - 1);
        sizes.resize(n, 1); // original points all have size 1
        Self { n, merges: Vec::with_capacity(n - 1), sizes }
    }

    pub(crate) fn get_size(&self, cid: usize) -> usize {
        // cid may refer to an original object (`cid < n`) or a merged
        // cluster; either way the vector has been sized appropriately.
        self.sizes[cid]
    }

    /// Record a new merge of clusters `a` and `b` at distance `dist`.
    ///
    /// Returns the identifier assigned to the newly-created cluster.
    pub(crate) fn add(&mut self, a: usize, dist: F, b: usize) -> usize {
        let size = self.get_size(a) + self.get_size(b);
        let new_id = self.n + self.merges.len();
        if new_id >= self.sizes.len() {
            self.sizes.push(size);
        } else {
            self.sizes[new_id] = size;
        }
        self.merges.push(Merge { idx1: a, idx2: b, distance: dist, size });
        new_id
    }

    /// Consume the builder and return the collected merge history.
    pub(crate) fn into_merges(self) -> MergeHistory<F> { self.merges }
}

/// Builder variant that additionally tracks prototypes for each merge.
pub(crate) struct PrototypeBuilder<F: Float> {
    n: usize,
    merges: PrototypeMergeHistory<F>,
    sizes: Vec<usize>,
}

impl<F: Float> PrototypeBuilder<F> {
    pub(crate) fn new(n: usize) -> Self {
        let mut sizes = Vec::with_capacity(2 * n - 1);
        sizes.resize(n, 1);
        Self { n, merges: Vec::with_capacity(n - 1), sizes }
    }

    pub(crate) fn get_size(&self, cid: usize) -> usize { self.sizes[cid] }

    pub(crate) fn add(&mut self, a: usize, dist: F, b: usize, prototype: Option<usize>) -> usize {
        let size = self.get_size(a) + self.get_size(b);
        let new_id = self.n + self.merges.len();
        if new_id >= self.sizes.len() {
            self.sizes.push(size);
        } else {
            self.sizes[new_id] = size;
        }
        self.merges.push(PrototypeMerge { idx1: a, idx2: b, distance: dist, size, prototype });
        new_id
    }

    pub(crate) fn into_merges(self) -> PrototypeMergeHistory<F> { self.merges }
}

impl<F: Float> AgglomerativeBuilder<F> for Builder<F> {
    type Output = MergeHistory<F>;

    fn new(n: usize) -> Self { Self::new(n) }

    fn get_size(&self, cid: usize) -> usize { self.get_size(cid) }

    fn add(&mut self, a: usize, dist: F, b: usize, _prototype: Option<usize>) -> usize {
        self.add(a, dist, b)
    }

    fn into_merges(self) -> Self::Output { self.into_merges() }
}

impl<F: Float> AgglomerativeBuilder<F> for PrototypeBuilder<F> {
    type Output = PrototypeMergeHistory<F>;

    fn new(n: usize) -> Self { Self::new(n) }

    fn get_size(&self, cid: usize) -> usize { self.get_size(cid) }

    fn add(&mut self, a: usize, dist: F, b: usize, prototype: Option<usize>) -> usize {
        self.add(a, dist, b, prototype)
    }

    fn into_merges(self) -> Self::Output { self.into_merges() }
}
