mod aknn;
mod construct;
mod knn;
mod priority;
mod range;

pub use construct::CoverTree;
pub use priority::CoverTreePrioritySearcher;

use crate::{DistPair, DistanceSearch, Float, KnnSearch, PrioritySearcherFactory, RangeSearch};

impl<F: Float, Q: DistanceSearch<F> + ?Sized> KnnSearch<F, Q> for CoverTree<F> {
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> { self.search_knn(query, k) }
}

impl<F: Float, Q: DistanceSearch<F> + ?Sized> RangeSearch<F, Q> for CoverTree<F> {
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        let mut result = Vec::new();
        self.search_range(query, radius, |pair| result.push(pair));
        result
    }
}

impl<F: Float, Q> PrioritySearcherFactory<F, Q> for CoverTree<F>
where
    Q: DistanceSearch<F> + ?Sized,
{
    type Searcher<'a>
        = CoverTreePrioritySearcher<'a, F>
    where
        Q: 'a,
        F: 'a;

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a,
    {
        CoverTreePrioritySearcher::new(self)
    }
}

/// Expansion heuristic from intrinsic dimensionality.
///
/// 2^(1 / sqrt(intrinsic_dim)).
pub fn expansion_heuristic_from_id(intrinsic_dim: f64) -> f64 {
    assert!(
        intrinsic_dim.is_finite() && intrinsic_dim > 0.0,
        "intrinsic_dim must be positive finite"
    );
    1.5_f64.powf(1.0 / intrinsic_dim.sqrt())
}
