//! Helper utilities used by the AGNES implementation.
//!
//! The algorithm relies on a small helper for building the merge history
//! while tracking cluster sizes, along with a convenience function for
//! computing indices into the condensed distance matrix.

use crate::DataAccess;
use num_traits::Float;

/// Compute the index in the condensed array for pair `(i, j)`, assuming
/// `i > j`.
///
/// This mirrors the inline helper formerly defined inside `agnes.rs`; we
/// pull it out here so that the core algorithm stays focused on the
/// clustering logic.
#[inline]
pub(crate) const fn triangle_index(i: usize, j: usize) -> usize {
    // i*(i-1)/2 + j
    // arithmetic is chosen so that it cannot overflow for reasonable sizes
    (i * (i - 1)) / 2 + j
}

#[inline]
pub(crate) fn condensed_get<F: Copy>(mat: &[F], a: usize, b: usize) -> F {
    if a > b {
        mat[triangle_index(a, b)]
    } else {
        mat[triangle_index(b, a)]
    }
}

#[inline]
pub(crate) fn condensed_set<F>(mat: &mut [F], a: usize, b: usize, value: F) {
    if a > b {
        mat[triangle_index(a, b)] = value;
    } else {
        mat[triangle_index(b, a)] = value;
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

pub(crate) fn initialize_distances<D: DataAccess>(data: &D) -> Vec<f64> {
    let n = data.size();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    for x in 1..n {
        for y in 0..x {
            distances.push(data.distance(x, y));
        }
    }
    distances
}

pub(crate) fn initialize_distances_and_prototypes<D: DataAccess>(
    data: &D,
) -> (Vec<f64>, Vec<usize>) {
    let n = data.size();
    let mut distances = Vec::with_capacity(n * (n - 1) / 2);
    let mut prototypes = Vec::with_capacity(n * (n - 1) / 2);
    for x in 1..n {
        for y in 0..x {
            distances.push(data.distance(x, y));
            prototypes.push(y);
        }
    }
    (distances, prototypes)
}

pub(crate) fn find_best_active_pair(
    distances: &[f64],
    clustermap: &[Option<usize>],
    end: usize,
) -> (usize, usize, f64) {
    let mut mindist = f64::INFINITY;
    let mut x = usize::MAX;
    let mut y = usize::MAX;
    for dx in 0..end {
        if clustermap[dx].is_none() {
            continue;
        }
        for dy in 0..dx {
            if clustermap[dy].is_none() {
                continue;
            }
            let dist = distances[triangle_index(dx, dy)];
            if dist < mindist {
                mindist = dist;
                x = dx;
                y = dy;
            }
        }
    }
    assert!(
        x != usize::MAX && y != usize::MAX,
        "no merge candidate found"
    );
    (x, y, mindist)
}

pub(crate) fn minimax_candidate<D: DataAccess>(
    data: &D,
    cx: &[usize],
    cy: &[usize],
) -> (f64, usize) {
    let mut best_dist = f64::INFINITY;
    let mut best_proto = cx[0];

    for &cand in cx.iter().chain(cy.iter()) {
        let mut max_dist = 0.0;
        for &p in cx {
            let d = data.distance(cand, p);
            if d > max_dist {
                max_dist = d;
                if max_dist >= best_dist {
                    break;
                }
            }
        }
        if max_dist >= best_dist {
            continue;
        }
        for &p in cy {
            let d = data.distance(cand, p);
            if d > max_dist {
                max_dist = d;
                if max_dist >= best_dist {
                    break;
                }
            }
        }
        if max_dist < best_dist {
            best_dist = max_dist;
            best_proto = cand;
        }
    }

    (best_dist, best_proto)
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
    pub prototype: usize,
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
        Self {
            n,
            merges: Vec::with_capacity(n - 1),
            sizes,
        }
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
        self.merges.push(Merge {
            idx1: a,
            idx2: b,
            distance: dist,
            size,
        });
        new_id
    }

    /// Consume the builder and return the collected merge history.
    pub(crate) fn into_merges(self) -> MergeHistory<F> {
        self.merges
    }
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
        Self {
            n,
            merges: Vec::with_capacity(n - 1),
            sizes,
        }
    }

    pub(crate) fn get_size(&self, cid: usize) -> usize {
        self.sizes[cid]
    }

    pub(crate) fn add(&mut self, a: usize, dist: F, b: usize, prototype: usize) -> usize {
        let size = self.get_size(a) + self.get_size(b);
        let new_id = self.n + self.merges.len();
        if new_id >= self.sizes.len() {
            self.sizes.push(size);
        } else {
            self.sizes[new_id] = size;
        }
        self.merges.push(PrototypeMerge {
            idx1: a,
            idx2: b,
            distance: dist,
            size,
            prototype,
        });
        new_id
    }

    pub(crate) fn into_merges(self) -> PrototypeMergeHistory<F> {
        self.merges
    }
}
