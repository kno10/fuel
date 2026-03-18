use crate::api::{Data, DistanceData, DistanceSearch, PointSearchData, VectorData};
use crate::distance::DistanceFunction;
use num_traits::Float;

// List of points with a distance function.
pub struct TableWithDistance<'a, T, DF: DistanceFunction<T, F>, F: Float> {
    data: &'a [T],
    distance_fn: DF,
    _phantom: std::marker::PhantomData<F>,
}

impl<'a, T, DF: DistanceFunction<T, F>, F: Float> Data for TableWithDistance<'a, T, DF, F> {
    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<'a, T, DF: DistanceFunction<T, F>, F> DistanceData<F> for TableWithDistance<'a, T, DF, F>
where
    DF: DistanceFunction<T, F>,
    F: Float,
{
    fn distance(&self, a: usize, b: usize) -> F {
        self.distance_fn.distance(&self.data[a], &self.data[b])
    }

    fn search_by_index(&self, idx: usize) -> impl DistanceSearch<F> {
        TableDistanceSearch {
            data: self,
            query: &self.data[idx],
        }
    }
}

impl<'a, T, DF, F> PointSearchData<F> for TableWithDistance<'a, T, DF, F>
where
    T: AsRef<[F]>,
    DF: DistanceFunction<T, F> + DistanceFunction<[F], F>,
    F: Float,
    TableWithDistance<'a, T, DF, F>: DistanceData<F>,
{
    fn search_by_point<'b>(&'b self, query: &'b [F]) -> impl DistanceSearch<F> + 'b {
        TablePointDistanceSearch { data: self, query }
    }
}

// TODO: Factor out a supertype with vector only, no distance.
impl<'a, T, DF, S> VectorData<S> for TableWithDistance<'a, T, DF, S>
where
    T: AsRef<[S]>,
    S: Copy + Float,
    DF: DistanceFunction<T, S>,
    TableWithDistance<'a, T, DF, S>: DistanceData<S>,
{
    fn dims(&self) -> usize {
        self.data
            .first()
            .map(|v| v.as_ref().len())
            .expect("An empty data set has no dimensionality.")
    }

    fn point(&self, idx: usize) -> &[S] {
        self.data[idx].as_ref()
    }
}

impl<'a, T, DF: DistanceFunction<T, F>, F: Float> TableWithDistance<'a, T, DF, F> {
    // FIXME: rename new
    pub const fn with_distance(data: &'a [T], distance_fn: DF) -> Self {
        Self {
            data,
            distance_fn,
            _phantom: std::marker::PhantomData,
        }
    }

    // TODO: this should go into some interface.
    pub const fn search_by_value(&'a self, query: &'a T) -> TableDistanceSearch<'a, T, DF, F> {
        TableDistanceSearch { data: self, query }
    }
}

///////////// Search wrapper
pub struct TableDistanceSearch<'a, T, DF: DistanceFunction<T, F>, F: Float> {
    data: &'a TableWithDistance<'a, T, DF, F>,
    query: &'a T,
}

impl<'a, T, DF: DistanceFunction<T, F>, F> DistanceSearch<F> for TableDistanceSearch<'a, T, DF, F>
where
    DF: DistanceFunction<T, F>,
    F: Float,
{
    fn query_distance(&self, b: usize) -> F {
        self.data
            .distance_fn
            .distance(self.query, &self.data.data[b])
    }
}

pub struct TablePointDistanceSearch<'a, T, DF, F>
where
    DF: DistanceFunction<T, F> + DistanceFunction<[F], F>,
    F: Float,
{
    data: &'a TableWithDistance<'a, T, DF, F>,
    query: &'a [F],
}

impl<'a, T, DF, F> DistanceSearch<F> for TablePointDistanceSearch<'a, T, DF, F>
where
    T: AsRef<[F]>,
    DF: DistanceFunction<T, F> + DistanceFunction<[F], F>,
    F: Float,
{
    fn query_distance(&self, b: usize) -> F {
        self.data
            .distance_fn
            .distance(self.query, self.data.data[b].as_ref())
    }
}
