use crate::{DataAccess, DistanceFunction};

pub struct MatrixDataAccess<'a, T, F = dyn DistanceFunction<T> + 'a> {
    data: &'a [T],
    distance_fn: F,
}

pub struct MatrixDataQuery<'a, 'q, T, F> {
    data_access: &'q MatrixDataAccess<'a, T, F>,
    query: &'q T,
}

impl<'a, T, F> MatrixDataAccess<'a, T, F> {
    pub const fn with_distance(data: &'a [T], distance_fn: F) -> Self {
        Self { data, distance_fn }
    }

    pub const fn with_query<'q>(&'q self, query: &'q T) -> MatrixDataQuery<'a, 'q, T, F> {
        MatrixDataQuery {
            data_access: self,
            query,
        }
    }

    pub fn with_query_index<'q>(&'q self, idx: usize) -> MatrixDataQuery<'a, 'q, T, F> {
        MatrixDataQuery {
            data_access: self,
            query: &self.data[idx],
        }
    }
}

impl<T, F> DataAccess for MatrixDataAccess<'_, T, F>
where
    F: DistanceFunction<T>,
{
    fn distance(&self, a: usize, b: usize) -> f64 {
        self.distance_fn.distance(&self.data[a], &self.data[b])
    }

    fn query_distance(&self, _b: usize) -> f64 {
        panic!("Query not set. Use with_query(...) or with_query_index(...)");
    }

    fn size(&self) -> usize {
        self.data.len()
    }
}

impl<T, F> DataAccess for MatrixDataQuery<'_, '_, T, F>
where
    F: DistanceFunction<T>,
{
    fn distance(&self, a: usize, b: usize) -> f64 {
        self.data_access
            .distance_fn
            .distance(&self.data_access.data[a], &self.data_access.data[b])
    }

    fn query_distance(&self, b: usize) -> f64 {
        self.data_access
            .distance_fn
            .distance(self.query, &self.data_access.data[b])
    }

    fn size(&self) -> usize {
        self.data_access.data.len()
    }
}
