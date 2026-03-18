mod construct;
mod knn_search;
mod priority_search;
mod range_search;

use num_traits::Float;

pub use priority_search::{NodePoints, PrioritySearcher, SearchFilter};

#[allow(non_camel_case_types)]
pub(crate) type vpsize = u32;



impl<F: Float, D: crate::DistanceData<F>> crate::KnnSearch<F, D> for VPTree<F> {
    fn search_knn_by_index(&self, data: &D, query_idx: usize, k: usize) -> Vec<crate::DistPair<F>> {
        self.search_knn(&data.search_by_index(query_idx), k)
    }
}

impl<F: Float, D: crate::DistanceData<F> + crate::VectorData<F> + ?Sized> crate::RangeSearch<F, D>
    for VPTree<F>
{
    fn search_range_by_index(&self, data: &D, query_idx: usize, radius: F) -> Vec<crate::DistPair<F>> {
        let mut result = Vec::new();
        self.search_range(&data.search_by_index(query_idx), radius, |pair| {
            result.push(pair);
        });
        result
    }
}

/// Priority searcher wrapper that binds a query point to a VP-tree search.
pub struct VPTreePrioritySearcher<'a, F, D>
where
    F: Float,
    D: crate::PointSearchData<F> + crate::VectorData<F> + ?Sized,
{
    inner: PrioritySearcher<'a, F>,
    data: &'a D,
    query: Vec<F>,
}

impl<'a, F, D> VPTreePrioritySearcher<'a, F, D>
where
    F: Float,
    D: crate::PointSearchData<F> + crate::VectorData<F> + ?Sized,
{
    pub fn new(tree: &'a VPTree<F>, data: &'a D, query: &[F]) -> Self {
        Self {
            inner: PrioritySearcher::new(tree),
            data,
            query: query.to_vec(),
        }
    }
}

impl<'a, F, D> crate::PrioritySearcherCore<F> for VPTreePrioritySearcher<'a, F, D>
where
    F: Float,
    D: crate::PointSearchData<F> + crate::VectorData<F> + ?Sized,
{
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn set_query<'b, DD: crate::DistanceData<F> + ?Sized>(&mut self, data: &'b DD, query: &[F]) {
        let _ = data;
        self.query.clear();
        self.query.extend_from_slice(query);
        self.reset();
    }

    fn next(&mut self) -> Option<crate::DistPair<F>> {
        self.inner.next(&self.data.search_by_point(&self.query))
    }

    fn all_lower_bound(&self) -> F {
        self.inner.all_lower_bound()
    }

    fn decrease_cutoff(&mut self, threshold: F) {
        self.inner.decrease_cutoff(threshold);
    }
}

impl<F: Float, D: crate::PointSearchData<F> + crate::VectorData<F> + ?Sized>
    crate::PrioritySearch<F, D> for VPTree<F>
{
    type Searcher<'a> = VPTreePrioritySearcher<'a, F, D>
    where
        F: 'a,
        D: 'a;

    fn priority_searcher<'a>(&'a self, data: &'a D, query: &'a [F]) -> Self::Searcher<'a> {
        VPTreePrioritySearcher::new(self, data, query)
    }
}

/// A candidate returned by the priority searcher along with its lower bound.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchCandidate<F> {
    pub(crate) distance: F,
    pub(crate) lower_bound: F,
    pub(crate) index: vpsize,
}

impl<F> SearchCandidate<F> {
    pub(crate) fn new(distance: F, lower_bound: F, index: usize) -> Self {
        Self {
            distance,
            lower_bound,
            index: index as vpsize,
        }
    }

    /// Distance of this candidate to the query point.
    #[must_use]
    pub fn distance(&self) -> F
    where
        F: Copy,
    {
        self.distance
    }

    /// Lower bound distance of this candidate to the query point.
    #[must_use]
    pub fn lower_bound(&self) -> F
    where
        F: Copy,
    {
        self.lower_bound
    }

    /// Index of this candidate in the backing data set.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index as usize
    }
}


#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct Bounds<F> {
    pub(crate) lower: F,
    pub(crate) upper: F,
}

impl<F> Bounds<F> {
    pub(crate) const fn new(lower: F, upper: F) -> Self {
        Self { lower, upper }
    }
}

pub struct VPTree<F> {
    points: Vec<vpsize>,
    bounds: Vec<Bounds<F>>,
}

#[cfg(test)]
mod tests;
