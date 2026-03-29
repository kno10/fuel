use crate::api::{Data, DistanceData, DistanceSearch, VectorData};
use crate::distance::{DistanceFunction, PartialDistance};
use crate::{CoordinateQuery, CoordinateSearch, Float, IndexQuery};

// List of points with a distance function.
pub struct TableWithDistance<
    'a,
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F>,
    F: Float = C,
> {
    data: &'a [T],
    distance_fn: DF,
    _coordinate_type: std::marker::PhantomData<C>,
    _distance_type: std::marker::PhantomData<F>,
}

impl<'a, C: Float, T: AsRef<[C]>, DF: DistanceFunction<[C], F>, F: Float> Data
    for TableWithDistance<'a, C, T, DF, F>
{
    fn size(&self) -> usize { self.data.len() }
}

impl<'a, C, T, DF, F> DistanceData<F> for TableWithDistance<'a, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F>,
    F: Float,
{
    type Query<'b>
        = TableQuery<'b, 'a, C, T, DF, F>
    where
        Self: 'b;

    fn distance(&self, a: usize, b: usize) -> F {
        self.distance_fn.distance(self.data[a].as_ref(), self.data[b].as_ref())
    }

    fn query(&self) -> Self::Query<'_> { TableQuery::new(self) }
}

// TODO: Factor out a supertype with vector only, no distance.
impl<'a, C, T, DF, F> VectorData<C> for TableWithDistance<'a, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F>,
    F: Float,
    TableWithDistance<'a, C, T, DF, F>: DistanceData<F>,
{
    fn dims(&self) -> usize {
        self.data
            .first()
            .map(|v| v.as_ref().len())
            .expect("an empty data set has no dimensionality")
    }

    fn point(&self, idx: usize) -> &[C] { self.data[idx].as_ref() }
}

impl<'a, C: Float, T: AsRef<[C]>, DF: DistanceFunction<[C], F>, F: Float>
    TableWithDistance<'a, C, T, DF, F>
{
    pub const fn with_distance(data: &'a [T], distance_fn: DF) -> Self {
        Self {
            data,
            distance_fn,
            _coordinate_type: std::marker::PhantomData,
            _distance_type: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TableQueryMode {
    Index(usize),
    Coordinates,
}

pub struct TableQuery<'q, 'd, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F>,
    F: Float,
{
    data: &'q TableWithDistance<'d, C, T, DF, F>,
    mode: TableQueryMode,
    coords: Vec<C>,
}

impl<'q, 'd, C, T, DF, F> TableQuery<'q, 'd, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F>,
    F: Float,
{
    fn new(data: &'q TableWithDistance<'d, C, T, DF, F>) -> Self {
        Self { data, mode: TableQueryMode::Index(0), coords: Vec::new() }
    }

    fn query_coords(&self) -> &[C] {
        match self.mode {
            TableQueryMode::Index(idx) => self.data.point(idx),
            TableQueryMode::Coordinates => &self.coords,
        }
    }
}

impl<'q, 'd, C, T, DF, F> DistanceSearch<F> for TableQuery<'q, 'd, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F>,
    F: Float,
{
    fn query_distance(&self, b: usize) -> F {
        debug_assert!(b < self.data.size());
        match self.mode {
            TableQueryMode::Index(idx) => self.data.distance(idx, b),
            TableQueryMode::Coordinates => {
                self.data.distance_fn.distance(self.coords.as_slice(), self.data.data[b].as_ref())
            }
        }
    }
}

impl<'q, 'd, C, T, DF, F> IndexQuery<F> for TableQuery<'q, 'd, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F>,
    F: Float,
{
    fn set_index(&mut self, idx: usize) {
        debug_assert!(idx < self.data.size());
        self.mode = TableQueryMode::Index(idx);
    }
}

impl<'q, 'd, C, T, DF, F> CoordinateSearch<C, F> for TableQuery<'q, 'd, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F> + PartialDistance<C, F>,
    F: Float,
{
    fn dims(&self) -> usize { self.data.dims() }

    fn query_coordinate(&self, axis: usize) -> C { self.query_coords()[axis] }

    fn delta_to_distance(&self, delta: C) -> F { self.data.distance_fn.axis_distance(delta) }

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

impl<'q, 'd, C, T, DF, F> CoordinateQuery<C, F> for TableQuery<'q, 'd, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: DistanceFunction<[C], F> + PartialDistance<C, F>,
    F: Float,
{
    fn set_coordinates(&mut self, coords: &[C]) {
        if self.data.size() > 0 {
            debug_assert_eq!(coords.len(), self.data.dims());
        }
        self.coords.clear();
        self.coords.extend_from_slice(coords);
        self.mode = TableQueryMode::Coordinates;
    }
}
