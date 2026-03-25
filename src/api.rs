use std::cmp::Ordering;

/// Common Float requirements to keep the source readable...
pub trait Float:
    num_traits::Float
    + Default
    + Copy // or Clone?
    + num_traits::AsPrimitive<Self>
    + num_traits::ToPrimitive
    + for<'a> std::ops::AddAssign<&'a Self>
    + for<'a> std::ops::MulAssign<&'a Self>
    + for<'a> std::ops::SubAssign<&'a Self>
    + for<'a> std::ops::DivAssign<&'a Self>
    + num_traits::MulAdd<Output = Self>
    + std::iter::Sum
    + num_traits::FromPrimitive
    + std::marker::Unpin
{
    fn cast<T: num_traits::NumCast>(x: T) -> Self {
        num_traits::NumCast::from(x).unwrap()
    }

    /// Convert this value to another float type.
    fn to_float<T: Float>(self) -> T {
        num_traits::cast(self).unwrap_or_else(|| {
            T::from_f64(self.to_f64().unwrap()).unwrap()
        })
    }
}

impl<
    T: num_traits::Float
        + Default
        + Copy // or Clone?
        + num_traits::AsPrimitive<T>
        + num_traits::ToPrimitive
        + for<'a> std::ops::AddAssign<&'a Self>
        + for<'a> std::ops::MulAssign<&'a Self>
        + for<'a> std::ops::SubAssign<&'a Self>
        + for<'a> std::ops::DivAssign<&'a Self>
        + num_traits::MulAdd<Output = Self>
        + std::iter::Sum
        + num_traits::FromPrimitive
        + std::marker::Unpin,
> Float for T
{
}

/// Toplevel data abstraction, only has a length.
pub trait Data {
    /// Get the size of the data set.
    fn size(&self) -> usize;

    /// Iterate points
    fn iter(&self) -> impl Iterator<Item = usize> { 0..self.size() }
}

/// Interface into a data set for distance calculations.
///
/// API for pairwise distances (computed or precomputed matrix).
pub trait DistanceData<F: Float>: Data {
    /// Query object produced by this data set.
    type Query<'a>: IndexQuery<F> + 'a
    where
        Self: 'a;

    /// Distance between two indexed points; must be symmetric.
    fn distance(&self, a: usize, b: usize) -> F;

    /// Create a reusable query object for this data set.
    fn query(&self) -> Self::Query<'_>;
}

/// A query that can be updated to point at different dataset indices.
pub trait IndexQuery<F: Float>: DistanceSearch<F> {
    /// Update the query to use the given dataset index.
    fn set_index(&mut self, idx: usize);

    /// Update the query and return it for chaining.
    fn with_index(mut self, idx: usize) -> Self
    where
        Self: Sized,
    {
        self.set_index(idx);
        self
    }
}

/// A query that can be updated to use explicit coordinates.
pub trait CoordinateQuery<C: Float, F: Float>: DistanceSearch<F> + CoordinateSearch<C, F> {
    /// Update the query to use the given coordinates.
    fn set_coordinates(&mut self, coords: &[C]);

    /// Update the query and return it for chaining.
    fn with_coordinates(mut self, coords: &[C]) -> Self
    where
        Self: Sized,
    {
        self.set_coordinates(coords);
        self
    }
}

/// Interface for data sets that support coordinate queries.
pub trait PointSearchData<C: Float, F: Float>: VectorData<C> + DistanceData<F>
where
    for<'a> Self::Query<'a>: CoordinateQuery<C, F>,
{
}

/// Interface for a running search
pub trait DistanceSearch<F: Float> {
    /// Distance from the (fixed) query point.
    fn query_distance(&self, b: usize) -> F;
}

/// Coordinate-base search interface, for k-d-tree etc.
pub trait CoordinateSearch<C: Float, F: Float> {
    /// Get the query coordinate for a single axis.
    fn query_coordinate(&self, axis: usize) -> C;

    /// Distance bound from a coordinate delta.
    fn delta_to_distance(&self, delta: C) -> F;

    /// Combine two axis bound distances.
    fn combine_axis_distances(&self, a: F, b: F) -> F;
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
    pub fn new(distance: F, index: usize) -> Self { Self { distance, index } }
}

impl<F: Float> DistPair<F> {
    /// An undefined value representing an empty candidate.
    ///
    /// Used by algorithms that need a placeholder for "no neighbor yet".
    pub fn undefined() -> Self { Self { distance: F::infinity(), index: usize::MAX } }

    /// Returns `true` if this is the sentinel value.
    pub fn is_sentinel(&self) -> bool { self.index == usize::MAX }
}

impl<F: Float> Eq for DistPair<F> {}

impl<F: Float> PartialOrd for DistPair<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<F: Float> Ord for DistPair<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed order
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.index.cmp(&self.index))
    }
}

/// Generic kNN search capability.
///
/// This trait allows algorithms (e.g. outlier detection) to be written once and
/// used with multiple underlying kNN indices (KD-tree, VP-tree, etc.).
pub trait KnnSearch<F: Float, Q: DistanceSearch<F> + ?Sized> {
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>>;
}

/// Generic range search capability.
pub trait RangeSearch<F: Float, Q: DistanceSearch<F> + ?Sized> {
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>>;
}

/// FIXME: can we use a trait like an Iter<u32> instead?
/// A view of a single index-node’s contents during a priority search.
///
/// This type is intentionally minimal and fast to construct. It is used by
/// `SearchFilter` implementations to inspect which dataset points are contained
/// within a node of a tree.
pub struct NodePoints<'a> {
    points: &'a [u32],
}

impl<'a> NodePoints<'a> {
    pub const fn new(points: &'a [u32]) -> Self { Self { points } }

    /// Iterate over dataset indices stored in this node.
    #[must_use]
    pub fn indices(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        self.points.iter().map(|&point| point as usize)
    }

    /// Number of points covered by this node.
    #[must_use]
    pub const fn len(&self) -> usize { self.points.len() }

    /// Whether this node covers no points.
    #[must_use]
    pub const fn is_empty(&self) -> bool { self.points.is_empty() }

    /// Dataset index of the vantage point (first element) of this node.
    #[must_use]
    pub fn first_index(&self) -> usize { self.points[0] as usize }
}

/// Filter consulted by priority searchers to prune results.
///
/// This trait is intended to be implemented by algorithms that wish to skip
/// whole subtrees or individual pivot points while scanning the search tree.
pub trait SearchFilter {
    /// Return `true` to skip this entire node and all of its descendants.
    fn skip_node(&mut self, _points: NodePoints<'_>) -> bool { false }

    /// Return `true` to skip the pivot point for the current node.
    fn skip_point(&mut self, index: usize) -> bool;
}

/// Priority searcher interface for incremental neighbor enumeration.
///
/// This is the in‑query searcher object (it yields `DistPair` candidates).
/// See `PrioritySearcherFactory` for how to obtain a searcher from an index.
pub trait PrioritySearcher<F: Float, Q: DistanceSearch<F> + ?Sized> {
    /// Reset the searcher state (e.g. to restart the same query from scratch).
    fn reset(&mut self);

    /// Reset this searcher and initialize the current cutoff and skip thresholds.
    fn reset_with_limits(&mut self, cutoff: F, skip: F);

    /// Yield the next candidate for the given query.
    fn next(&mut self, query: &Q) -> Option<crate::DistPair<F>>;

    /// Yield the next candidate for the given query, consulting a filter to prune results.
    fn next_with_filter<S>(&mut self, query: &Q, filter: &mut S) -> Option<crate::DistPair<F>>
    where
        S: SearchFilter;

    /// Lower bound for remaining candidates.
    fn all_lower_bound(&self) -> F;

    /// Reduce the upper cutoff threshold.
    fn decrease_cutoff(&mut self, threshold: F);
}

/// Generic factory for creating a priority searcher.
///
/// This trait is implemented by index types (e.g. `VPTree`, `KdTree`) that
/// can produce an initially-unconfigured searcher instance.
pub trait PrioritySearcherFactory<F: Float, Q: DistanceSearch<F> + ?Sized> {
    type Searcher<'a>: PrioritySearcher<F, Q>
    where
        Self: 'a,
        Q: 'a,
        F: 'a; // FIXME: do we need these lifetimes?

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a;

    // FIXME: add priority_search helper methods once query factories are stabilized.
}

/// Access into individual points when coordinate data is available.
pub trait VectorData<C>: Data {
    /// Number of dimensions for every point.
    fn dims(&self) -> usize;

    /// Returns a slice covering the point at `idx`.
    fn point(&self, idx: usize) -> &[C];

    // TODO: also allow direct access to single coordinates?
}

// blanket implementations for references so borrowed datasets also satisfy traits
impl<D: Data> Data for &D {
    fn size(&self) -> usize { (*self).size() }
}

impl<D, F> DistanceData<F> for &D
where
    D: DistanceData<F>,
    F: Float,
{
    type Query<'a>
        = D::Query<'a>
    where
        Self: 'a;

    fn distance(&self, a: usize, b: usize) -> F { (*self).distance(a, b) }

    fn query(&self) -> Self::Query<'_> { (*self).query() }
}

// blanket impl so that references to search objects also satisfy DistanceSearch
impl<D, F> DistanceSearch<F> for &D
where
    D: DistanceSearch<F>,
    F: Float,
{
    fn query_distance(&self, b: usize) -> F { (*self).query_distance(b) }
}

impl<C, D, F> CoordinateSearch<C, F> for &D
where
    C: Float,
    D: CoordinateSearch<C, F>,
    F: Float,
{
    fn query_coordinate(&self, axis: usize) -> C { (*self).query_coordinate(axis) }

    fn delta_to_distance(&self, delta: C) -> F { (*self).delta_to_distance(delta) }

    fn combine_axis_distances(&self, a: F, b: F) -> F { (*self).combine_axis_distances(a, b) }
}

// Allow boxed `DistanceSearch` trait objects to satisfy the trait itself.
impl<'a, F: Float> DistanceSearch<F> for Box<dyn DistanceSearch<F> + 'a> {
    fn query_distance(&self, b: usize) -> F { (**self).query_distance(b) }
}

impl<'a, C: Float, F: Float> CoordinateSearch<C, F> for Box<dyn CoordinateSearch<C, F> + 'a> {
    fn query_coordinate(&self, axis: usize) -> C { (**self).query_coordinate(axis) }

    fn delta_to_distance(&self, delta: C) -> F { (**self).delta_to_distance(delta) }

    fn combine_axis_distances(&self, a: F, b: F) -> F { (**self).combine_axis_distances(a, b) }
}

impl<C, D, F> PointSearchData<C, F> for D
where
    C: Float,
    D: VectorData<C> + DistanceData<F>,
    F: Float,
    for<'a> D::Query<'a>: CoordinateQuery<C, F>,
{
}

impl<D, F> VectorData<F> for &D
where
    D: VectorData<F>,
{
    fn dims(&self) -> usize { (*self).dims() }

    fn point(&self, idx: usize) -> &[F] { (*self).point(idx) }
}
