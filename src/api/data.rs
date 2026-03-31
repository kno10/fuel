use crate::api::float::Float;
use crate::api::query::{CoordinateQuery, IndexQuery};

/// Toplevel data abstraction, only has a length.
pub trait Data {
    /// Number of points in the data set.
    fn len(&self) -> usize;

    /// Equivalent to `len() == 0`.
    fn is_empty(&self) -> bool { self.len() == 0 }

    /// Iterate point indices from 0..len().
    fn iter(&self) -> std::ops::Range<usize> { 0..self.len() }
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

/// Access into individual points when coordinate data is available.
pub trait VectorData<C>: Data {
    /// Number of input points.
    fn nrows(&self) -> usize { self.len() }

    /// Number of dimensions for every point.
    fn ncols(&self) -> usize { self.dims() }

    /// Number of dimensions for every point.
    fn dims(&self) -> usize;

    /// Returns a slice covering the point at `idx`.
    fn point(&self, idx: usize) -> &[C];

    /// Copy a point into a caller-provided scratch buffer.
    fn load_into(&self, idx: usize, out: &mut [C], d: usize)
    where
        C: Copy,
    {
        debug_assert_eq!(d, self.dims());
        out[..d].copy_from_slice(self.point(idx));
    }

    // TODO: also allow direct access to single coordinates?
}

/// Interface for data sets that support coordinate queries.
pub trait PointSearchData<C: Float, F: Float>: VectorData<C> + DistanceData<F>
where
    for<'a> Self::Query<'a>: CoordinateQuery<C, F>,
{
}

// blanket implementations for references so borrowed datasets also satisfy traits
impl<D: Data> Data for &D {
    fn len(&self) -> usize { (*self).len() }
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
