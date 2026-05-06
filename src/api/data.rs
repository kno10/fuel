use ndarray::{Array2, ArrayView2, CowArray, Ix2};

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

    /// Whether the distances are already squared Euclidean values.
    fn is_squared_distance(&self) -> bool { false }
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

    /// Optional direct access to the underlying data as an ndarray view.
    ///
    /// Implementations may return `None` if a contiguous 2-D ndarray view is
    /// not available.  This is a performance hint for algorithms that can
    /// avoid an explicit copy when the data is already ndarray-backed.
    fn as_ndarray(&self) -> Option<ArrayView2<'_, C>> { None }

    /// Returns the data as a C-contiguous 2-D ndarray, borrowing if the
    /// underlying storage is already standard-layout, or allocating a flat
    /// copy otherwise.
    fn to_ndarray(&self) -> CowArray<'_, C, Ix2>
    where
        C: Copy + Default,
    {
        if let Some(v) = self.as_ndarray() && v.is_standard_layout() {
            return v.into();
        }
        let (n, d) = (self.nrows(), self.dims());
        let mut buf = vec![C::default(); n * d];
        for i in 0..n {
            self.load_into(i, &mut buf[i * d..(i + 1) * d], d);
        }
        Array2::from_shape_vec((n, d), buf).expect("flat buffer shape mismatch").into()
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

    fn is_squared_distance(&self) -> bool { (*self).is_squared_distance() }
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
