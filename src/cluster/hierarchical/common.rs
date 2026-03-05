//! Helper utilities used by the AGNES implementation.
//!
//! The algorithm relies on a small helper for building the merge history
//! while tracking cluster sizes, along with a convenience function for
//! computing indices into the condensed distance matrix.

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

use num_traits::Float;

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
