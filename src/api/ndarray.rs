use ndarray::{ArrayBase, Data as NdData, Ix2, RawData};

use crate::distance::{DistanceFunction, Euclidean, PartialDistance};
use crate::{
    CoordinateQuery, CoordinateSearch, Data, DistanceData, DistanceSearch, Float, IndexQuery,
    VectorData as Dataset,
};

/// Wrapper to use ndarrays as data sets
pub struct NdArrayDataset<'a, N: 'a, A: 'a> {
    data: &'a A,
    marker: std::marker::PhantomData<&'a N>,
}
impl<'a, N, A> NdArrayDataset<'a, N, A> {
    /// Borrow array
    pub fn new(data: &'a A) -> Self {
        // TODO: log a warning if the data is not in standard layout?
        Self { data, marker: std::marker::PhantomData {} }
    }
}
impl<N, D> Data for NdArrayDataset<'_, N, ArrayBase<D, Ix2>>
where
    N: Copy,
    D: RawData<Elem = N> + NdData,
{
    fn len(&self) -> usize { self.data.nrows() }
}

impl<N, D> Dataset<N> for NdArrayDataset<'_, N, ArrayBase<D, Ix2>>
where
    N: Copy,
    D: RawData<Elem = N> + NdData,
{
    fn dims(&self) -> usize { self.data.ncols() }

    fn point(&self, idx: usize) -> &[N] {
        let row = self.data.row(idx);
        unsafe { std::slice::from_raw_parts(row.as_ptr(), self.dims()) }
    }

    fn load_into(&self, i: usize, vec: &mut [N], d: usize) {
        debug_assert_eq!(d, self.dims(), "Dimensionality mismatch");
        let ndrow = self.data.row(i);
        if let Some(s) = ndrow.as_slice() {
            vec.copy_from_slice(s);
        } else {
            // Fallback, not contiguous
            for i in 0..d {
                unsafe {
                    *vec.get_unchecked_mut(i) = *ndrow.uget(i);
                }
            }
        }
    }

    fn as_ndarray(&self) -> Option<ndarray::ArrayView2<'_, N>> { Some(self.data.view()) }
}

pub struct NdArrayDatasetWithDistance<'a, N: 'a, A: 'a, DF, F = N> {
    data: &'a A,
    distance_fn: DF,
    _coordinate_type: std::marker::PhantomData<&'a N>,
    _distance_type: std::marker::PhantomData<F>,
}

impl<'a, N, A, DF, F> NdArrayDatasetWithDistance<'a, N, A, DF, F> {
    pub fn with_distance(data: &'a A, distance_fn: DF) -> Self {
        Self {
            data,
            distance_fn,
            _coordinate_type: std::marker::PhantomData,
            _distance_type: std::marker::PhantomData,
        }
    }
}

impl<'a, N, A> NdArrayDatasetWithDistance<'a, N, A, Euclidean, N>
where
    N: Float,
{
    pub fn new(data: &'a A) -> Self { Self::with_distance(data, Euclidean) }
}

impl<N, D, DF, F> Data for NdArrayDatasetWithDistance<'_, N, ArrayBase<D, Ix2>, DF, F>
where
    N: Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    fn len(&self) -> usize { self.data.nrows() }
}

impl<N, D, DF, F> Dataset<N> for NdArrayDatasetWithDistance<'_, N, ArrayBase<D, Ix2>, DF, F>
where
    N: Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    fn dims(&self) -> usize { self.data.ncols() }

    fn point(&self, idx: usize) -> &[N] {
        let row = self.data.row(idx);
        unsafe { std::slice::from_raw_parts(row.as_ptr(), self.dims()) }
    }

    fn load_into(&self, i: usize, vec: &mut [N], d: usize) {
        debug_assert_eq!(d, self.dims(), "Dimensionality mismatch");
        let ndrow = self.data.row(i);
        if let Some(s) = ndrow.as_slice() {
            vec.copy_from_slice(s);
        } else {
            for i in 0..d {
                unsafe {
                    *vec.get_unchecked_mut(i) = *ndrow.uget(i);
                }
            }
        }
    }
}

pub struct NdArrayDistanceQuery<'a, N, D, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    data: &'a NdArrayDatasetWithDistance<'a, N, ArrayBase<D, Ix2>, DF, F>,
    index: usize,
    coords: Option<Vec<N>>,
}

impl<'a, N, D, DF, F> NdArrayDistanceQuery<'a, N, D, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    fn new(data: &'a NdArrayDatasetWithDistance<'a, N, ArrayBase<D, Ix2>, DF, F>) -> Self {
        Self { data, index: 0, coords: None }
    }
}

impl<'a, N, D, DF, F> DistanceSearch<F> for NdArrayDistanceQuery<'a, N, D, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    fn query_distance(&self, b: usize) -> F {
        let target = self.data.point(b);
        if let Some(coords) = &self.coords {
            self.data.distance_fn.distance(coords.as_ref(), target)
        } else {
            self.data.distance(self.index, b)
        }
    }
}

impl<'a, N, D, DF, F> NdArrayDistanceQuery<'a, N, D, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    fn query_coords(&self) -> &[N] {
        match &self.coords {
            Some(coords) => coords.as_slice(),
            None => self.data.point(self.index),
        }
    }
}

impl<'a, N, D, DF, F> CoordinateSearch<N, F> for NdArrayDistanceQuery<'a, N, D, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F> + PartialDistance<N, F>,
    F: Float,
{
    fn dims(&self) -> usize { self.data.dims() }

    fn query_coordinate(&self, axis: usize) -> N { self.query_coords()[axis] }

    fn delta_to_distance(&self, delta: N) -> F { self.data.distance_fn.axis_distance(delta) }

    fn distance_to_range_bound(&self, distance: F) -> F {
        self.data.distance_fn.distance_to_range_bound(distance)
    }

    fn range_bound_to_distance(&self, bound: F) -> F {
        self.data.distance_fn.range_bound_to_distance(bound)
    }

    fn replace_axis_distance(
        &self, current: F, axis: usize, old_axis: F, new_axis: F, axis_bounds: &[F],
    ) -> F {
        self.data.distance_fn.replace_axis_distance(current, axis, old_axis, new_axis, axis_bounds)
    }
}

impl<'a, N, D, DF, F> CoordinateQuery<N, F> for NdArrayDistanceQuery<'a, N, D, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F> + PartialDistance<N, F>,
    F: Float,
{
    fn set_coordinates(&mut self, coords: &[N]) {
        if self.data.len() > 0 {
            debug_assert_eq!(coords.len(), self.data.dims());
        }
        self.coords = Some(coords.to_vec());
    }
}

impl<'a, N, D, DF, F> IndexQuery<F> for NdArrayDistanceQuery<'a, N, D, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    fn set_index(&mut self, idx: usize) {
        self.index = idx;
        self.coords = None;
    }

    fn query_index(&self) -> usize { self.index }
}

impl<N, D, DF, F> DistanceData<F> for NdArrayDatasetWithDistance<'_, N, ArrayBase<D, Ix2>, DF, F>
where
    N: Float + Copy,
    D: RawData<Elem = N> + NdData,
    DF: DistanceFunction<[N], F>,
    F: Float,
{
    type Query<'b>
        = NdArrayDistanceQuery<'b, N, D, DF, F>
    where
        Self: 'b;

    fn distance(&self, a: usize, b: usize) -> F {
        self.distance_fn.distance(self.point(a), self.point(b))
    }

    fn query(&self) -> Self::Query<'_> { NdArrayDistanceQuery::new(self) }

    fn is_squared_distance(&self) -> bool { self.distance_fn.is_squared_distance() }
}
