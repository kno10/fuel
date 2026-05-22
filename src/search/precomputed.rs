use crate::DistanceSearch;
use crate::api::data::DistanceData;
use crate::api::float::Float;
use crate::api::parallel::ParMap;
use crate::api::query::IndexQuery;
use crate::api::search::{DistPair, KnnSearch, PrioritySearcherFactory, RangeSearch};

/// A wrapper that precomputes a kNN table for parameter sweeps.
///
/// The precomputed searcher computes kNN results up to `max_k` for every data
/// point in parallel. For requests with `k <= max_k`, it serves the result
/// directly from the precomputed table and preserves ties at the requested
/// k-th distance.
///
/// Larger k requests are forwarded to the original searcher.
pub struct PrecomputedKnnSearcher<F, S>
where
    F: Float,
    S: Sync,
{
    source: S,
    precomputed: Vec<Vec<DistPair<F>>>,
    max_k: usize,
}

impl<F, S> PrecomputedKnnSearcher<F, S>
where
    F: Float,
    S: Sync,
{
    /// Build a new precomputed searcher.
    ///
    /// - `source` is the original searcher used for delegated large-k queries.
    /// - `data` is the dataset whose indexed points will be precomputed.
    /// - `max_k` is the maximum k value expected in parameter sweeps.
    pub fn new<'a, D>(source: S, data: &'a D, max_k: usize) -> Self
    where
        D: DistanceData<F> + Sync,
        D::Query<'a>: IndexQuery<F> + Send,
        S: KnnSearch<F, D::Query<'a>> + Sync,
    {
        let n = data.len();
        let max_k = max_k.min(n);
        let precomputed = if n == 0 || max_k == 0 {
            Vec::new()
        } else {
            (0..n).par_map(|index| {
                let mut query = data.query();
                query.set_index(index);
                source.search_knn(&query, max_k)
            })
        };

        Self { source, precomputed, max_k }
    }

    /// Returns the maximum k precomputed by this proxy.
    pub fn max_k(&self) -> usize { self.max_k }

    fn subset_with_ties(&self, k: usize, results: &[DistPair<F>]) -> Vec<DistPair<F>> {
        if k == 0 || results.is_empty() {
            return Vec::new();
        }

        let kth_distance = results[k.min(results.len()) - 1].distance;
        results.iter().take_while(|pair| pair.distance <= kth_distance).copied().collect()
    }
}

impl<F, S, Q> KnnSearch<F, Q> for PrecomputedKnnSearcher<F, S>
where
    F: Float,
    Q: DistanceSearch<F> + IndexQuery<F> + Send + ?Sized,
    S: KnnSearch<F, Q> + Sync,
{
    fn search_knn(&self, query: &Q, k: usize) -> Vec<DistPair<F>> {
        if k == 0 {
            return Vec::new();
        }

        if k <= self.max_k && !self.precomputed.is_empty() {
            let index = query.query_index();
            if let Some(results) = self.precomputed.get(index) {
                return self.subset_with_ties(k, results);
            }
        }

        self.source.search_knn(query, k)
    }
}

impl<F, S, Q> RangeSearch<F, Q> for PrecomputedKnnSearcher<F, S>
where
    F: Float,
    Q: DistanceSearch<F> + IndexQuery<F> + Send + ?Sized,
    S: KnnSearch<F, Q> + RangeSearch<F, Q> + Sync,
{
    fn search_range(&self, query: &Q, radius: F) -> Vec<DistPair<F>> {
        self.source.search_range(query, radius)
    }
}

impl<F, S, Q> PrioritySearcherFactory<F, Q> for PrecomputedKnnSearcher<F, S>
where
    F: Float,
    Q: DistanceSearch<F> + ?Sized,
    S: PrioritySearcherFactory<F, Q> + Sync,
{
    type Searcher<'a>
        = <S as PrioritySearcherFactory<F, Q>>::Searcher<'a>
    where
        S: 'a,
        Q: 'a,
        F: 'a;

    fn priority_searcher<'a>(&'a self) -> Self::Searcher<'a>
    where
        Q: 'a,
    {
        self.source.priority_searcher()
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::api::data::Data;
    use crate::api::search::linear_scan_knn;
    use crate::api::tabular::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    #[test]
    fn precomputed_knn_searcher_reuses_precomputed_results() {
        let points =
            vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let table: TableWithDistance<_, _, _, f64> =
            TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree = VPTree::new(&table, 2, &mut rng);

        let precomputed = PrecomputedKnnSearcher::new(tree, &table, 2);

        for index in 0..table.len() {
            let mut query = table.query();
            query.set_index(index);

            let expected_row_k1 = linear_scan_knn(&table, &query, 1);
            let precomputed_result = precomputed.search_knn(&query, 1);
            assert!(precomputed_result.len() >= 1);
            let first_k_distance = precomputed_result[0].distance;
            assert!(precomputed_result.iter().skip(1).all(|pair| pair.distance == first_k_distance));
            assert_eq!(expected_row_k1, precomputed_result);

            let expected_row_k2 = linear_scan_knn(&table, &query, 2);
            let precomputed_result2 = precomputed.search_knn(&query, 2);
            assert!(precomputed_result2.len() >= 2);
            let second_k_distance = precomputed_result2[1].distance;
            assert!(precomputed_result2.iter().skip(2).all(|pair| pair.distance == second_k_distance));
            assert_eq!(expected_row_k2, precomputed_result2);
        }
    }
}
