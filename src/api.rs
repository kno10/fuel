use num_traits::Float;

/// Toplevel data abstraction, only has a length.
pub trait Data {
    /// Get the size of the data set.
    fn size(&self) -> usize;

    /// Iterate points
    fn iter(&self) -> impl Iterator<Item = usize> {
        0..self.size()
    }
}

/// Interface into a data set for distance calculations.
///
/// API for pairwise distances (computed or precomputed matrix).
pub trait DistanceData<F: Float>: Data {
    /// Distance between two indexed points; must be symmetric.
    fn distance(&self, a: usize, b: usize) -> F;

    /// Start a search from a query point index.
    fn search_by_index(&self, idx: usize) -> impl DistanceSearch<F>;
}

/// Interface into a data set that can be searched using an explicit query point.
pub trait PointSearchData<F: Float>: DistanceData<F> {
    /// Start a search from an explicit query point.
    fn search_by_point<'a>(&'a self, point: &'a [F]) -> impl DistanceSearch<F> + 'a;
}

/// Interface for a running search
pub trait DistanceSearch<F: Float> {
    /// Distance from the (fixed) query point.
    fn query_distance(&self, b: usize) -> F;
}

/// Simple pair of (distance, index) returned by search operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistPair<F> {
    /// Distance from the query point.
    pub distance: F,
    /// Index of the point in the data set.
    pub index: usize,
}

impl<F> DistPair<F> {
    /// Construct a new pair.
    pub fn new(distance: F, index: usize) -> Self {
        Self { distance, index }
    }
}

impl<F: Float> DistPair<F> {
    /// An undefined value representing an empty candidate.
    ///
    /// Used by algorithms that need a placeholder for "no neighbor yet".
    pub fn undefined() -> Self {
        Self {
            distance: F::infinity(),
            index: usize::MAX,
        }
    }

    /// Returns `true` if this is the sentinel value.
    pub fn is_sentinel(&self) -> bool {
        self.index == usize::MAX
    }
}

impl<F: Float> Eq for DistPair<F> {}

impl<F: Float> PartialOrd for DistPair<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Reverse the comparison so that the default `BinaryHeap` behaves as a
        // min-heap (smallest distance is considered "greatest").
        if let Some(ord) = other.distance.partial_cmp(&self.distance) {
            Some(ord.then_with(|| self.index.cmp(&other.index)))
        } else {
            Some(std::cmp::Ordering::Equal)
        }
    }
}

impl<F: Float> Ord for DistPair<F> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Generic kNN search capability.
///
/// This trait allows algorithms (e.g. outlier detection) to be written once and
/// used with multiple underlying kNN indices (KD-tree, VP-tree, etc.).
pub trait KnnSearch<F: Float, D: DistanceData<F> + ?Sized> {
    /// Find the `k` nearest neighbors to the point at `query_idx`.
    fn search_knn_by_index(&self, data: &D, query_idx: usize, k: usize) -> Vec<DistPair<F>>;
}

/// Generic range search capability.
pub trait RangeSearch<F: Float, D: DistanceData<F> + ?Sized> {
    /// Find all points within `radius` of the point at `query_idx`.
    fn search_range_by_index(&self, data: &D, query_idx: usize, radius: F) -> Vec<DistPair<F>>;
}

/// Priority searcher interface for incremental neighbor enumeration.
///
/// See `PrioritySearch` for how to obtain a searcher from an index.
pub trait PrioritySearcherCore<F: Float> {
    /// Reset the searcher state (e.g. to restart the same query from scratch).
    fn reset(&mut self);

    /// Update the query for this searcher.
    fn set_query<D: DistanceData<F> + ?Sized>(&mut self, data: &D, query: &[F]);

    /// Yield the next candidate.
    fn next(&mut self) -> Option<crate::DistPair<F>>;

    /// Lower bound for remaining candidates.
    fn all_lower_bound(&self) -> F;

    /// Reduce the upper cutoff threshold.
    fn decrease_cutoff(&mut self, threshold: F);
}

/// Generic factory for creating a priority searcher.
pub trait PrioritySearch<F: Float, D: DistanceData<F> + ?Sized> {
    type Searcher<'a>: PrioritySearcherCore<F>
    where
        Self: 'a,
        F: 'a,
        D: 'a;

    /// Create a new priority searcher for a given query.
    fn priority_searcher<'a>(&'a self, data: &'a D, query: &'a [F]) -> Self::Searcher<'a>;
}

/// Access into individual points when coordinate data is available.
pub trait VectorData<F>: Data {
    /// Number of dimensions for every point.
    fn dims(&self) -> usize;

    /// Returns a slice covering the point at `idx`.
    fn point(&self, idx: usize) -> &[F];

    // TODO: also allow direct access to single values?
}

// blanket implementations for references so borrowed datasets also satisfy traits
impl<D: Data> Data for &D {
    fn size(&self) -> usize {
        (*self).size()
    }
}

impl<D, F> DistanceData<F> for &D
where
    D: DistanceData<F>,
    F: Float,
{
    fn distance(&self, a: usize, b: usize) -> F {
        (*self).distance(a, b)
    }

    fn search_by_index(&self, idx: usize) -> impl DistanceSearch<F> {
        (*self).search_by_index(idx)
    }
}

// blanket impl so that references to search objects also satisfy DistanceSearch
impl<D, F> DistanceSearch<F> for &D
where
    D: DistanceSearch<F>,
    F: Float,
{
    fn query_distance(&self, b: usize) -> F {
        (*self).query_distance(b)
    }
}

impl<D, F> PointSearchData<F> for &D
where
    D: PointSearchData<F>,
    F: Float,
{
    fn search_by_point<'a>(&'a self, point: &'a [F]) -> impl DistanceSearch<F> + 'a {
        (*self).search_by_point(point)
    }
}

impl<D, F> VectorData<F> for &D
where
    D: VectorData<F>,
{
    fn dims(&self) -> usize {
        (*self).dims()
    }

    fn point(&self, idx: usize) -> &[F] {
        (*self).point(idx)
    }
}
