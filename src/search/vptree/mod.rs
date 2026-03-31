mod aknn;
mod construct;
mod knn;
mod priority;
mod range;

pub use priority::PrioritySearcher;

use crate::{DistPair, DistanceSearch, Float, KnnSearch, PrioritySearcherFactory, RangeSearch};
pub use crate::{NodePoints, SearchFilter};

#[allow(non_camel_case_types)]
pub(crate) type vpsize = u32;

impl<F: Float, Q: DistanceSearch<F> + ?Sized> KnnSearch<F, Q> for VPTree<F> {
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> { self.search_knn(query, k) }
}

impl<F: Float, Q: DistanceSearch<F> + ?Sized> RangeSearch<F, Q> for VPTree<F> {
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        let mut result = Vec::new();
        self.search_range(query, radius, |pair| {
            result.push(pair);
        });
        result
    }
}

/// Priority searcher wrapper that binds a query point to a VP-tree search.
pub struct VPTreePrioritySearcher<'a, F>
where
    F: Float,
{
    inner: PrioritySearcher<'a, F>,
}

impl<'a, F> VPTreePrioritySearcher<'a, F>
where
    F: Float,
{
    pub fn new(tree: &'a VPTree<F>) -> Self { Self { inner: PrioritySearcher::new(tree) } }
}

impl<'a, F, Q> crate::PrioritySearcher<F, Q> for VPTreePrioritySearcher<'a, F>
where
    F: Float,
    Q: DistanceSearch<F> + ?Sized,
{
    fn reset(&mut self) { self.inner.reset(); }

    fn reset_with_limits(&mut self, cutoff: F, skip: F) {
        self.inner.reset_with_limits(cutoff, skip);
    }

    fn next(&mut self, query: &Q) -> Option<DistPair<F>> { self.inner.next(query) }

    fn next_with_filter<S>(&mut self, query: &Q, filter: &mut S) -> Option<DistPair<F>>
    where
        S: SearchFilter,
    {
        self.inner.next_with_filter(query, filter)
    }

    fn all_lower_bound(&self) -> F { self.inner.all_lower_bound() }

    fn decrease_cutoff(&mut self, threshold: F) { self.inner.decrease_cutoff(threshold); }
}

impl<F: Float, Q> PrioritySearcherFactory<F, Q> for VPTree<F>
where
    Q: DistanceSearch<F> + ?Sized,
{
    type Searcher<'a>
        = VPTreePrioritySearcher<'a, F>
    where
        Q: 'a,
        F: 'a;

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a,
    {
        VPTreePrioritySearcher::new(self)
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
        Self { distance, lower_bound, index: index as vpsize }
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
    pub const fn index(&self) -> usize { self.index as usize }
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct Bounds<F> {
    pub(crate) lower: F,
    pub(crate) upper: F,
}

impl<F> Bounds<F> {
    pub(crate) const fn new(lower: F, upper: F) -> Self { Self { lower, upper } }
}

pub struct VPTree<F> {
    points: Vec<vpsize>,
    bounds: Vec<Bounds<F>>,
}

#[cfg(test)]
mod tests;
