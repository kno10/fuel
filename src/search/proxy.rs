use crate::api::data::DistanceData;
use crate::api::float::Float;
use crate::api::parallel::ParMap;
use crate::api::query::IndexQuery;
use crate::api::search::{DistPair, KnnSearch, RangeSearch};

/// A wrapper that precomputes a kNN table for parameter sweeps.
///
/// The proxy searcher computes kNN results up to `max_k` for every data
/// point in parallel. For requests with `k <= max_k`, it serves the result
/// directly from the precomputed table and preserves ties at the requested
/// k-th distance.
///
/// Larger k requests are forwarded to the original searcher.
pub struct ProxyKnnSearcher<'a, F, D, S>
where
    F: Float,
    D: DistanceData<F> + Sync,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    D::Query<'a>: IndexQuery<F> + Send,
{
    source: &'a S,
    precomputed: Vec<Vec<DistPair<F>>>,
    max_k: usize,
    _marker: std::marker::PhantomData<&'a D>,
}

impl<'a, F, D, S> ProxyKnnSearcher<'a, F, D, S>
where
    F: Float,
    D: DistanceData<F> + Sync,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    D::Query<'a>: IndexQuery<F> + Send,
{
    /// Build a new proxy searcher.
    ///
    /// - `source` is the original searcher used for delegated large-k queries.
    /// - `data` is the dataset whose indexed points will be precomputed.
    /// - `max_k` is the maximum k value expected in parameter sweeps.
    pub fn new(source: &'a S, data: &'a D, max_k: usize) -> Self {
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

        Self {
            source,
            precomputed,
            max_k,
            _marker: std::marker::PhantomData,
        }
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

impl<'a, F, D, S> KnnSearch<F, D::Query<'a>> for ProxyKnnSearcher<'a, F, D, S>
where
    F: Float,
    D: DistanceData<F> + Sync,
    S: KnnSearch<F, D::Query<'a>> + Sync,
    D::Query<'a>: IndexQuery<F> + Send,
{
    fn search_knn(&self, query: &D::Query<'a>, k: usize) -> Vec<DistPair<F>> {
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

impl<'a, F, D, S> RangeSearch<F, D::Query<'a>> for ProxyKnnSearcher<'a, F, D, S>
where
    F: Float,
    D: DistanceData<F> + Sync,
    S: KnnSearch<F, D::Query<'a>> + RangeSearch<F, D::Query<'a>> + Sync,
    D::Query<'a>: IndexQuery<F> + Send,
{
    fn search_range(&self, query: &D::Query<'a>, radius: F) -> Vec<DistPair<F>> {
        self.source.search_range(query, radius)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::api::data::Data;
    use crate::api::tabular::TableWithDistance;
    use crate::distance::Euclidean;
    use crate::search::vptree::VPTree;

    #[test]
    fn proxy_knn_searcher_reuses_precomputed_results() {
        let points =
            vec![vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
        let table: TableWithDistance<_, _, _, f64> =
            TableWithDistance::with_distance(&points, Euclidean);
        let mut rng = StdRng::seed_from_u64(42);
        let tree = VPTree::new(&table, 2, &mut rng);
        let proxy = ProxyKnnSearcher::new(&tree, &table, 2);

        for index in 0..table.len() {
            let mut query = table.query();
            query.set_index(index);

            let direct = tree.search_knn(&query, 1);
            let proxy_result = proxy.search_knn(&query, 1);
            assert_eq!(direct, proxy_result);

            let direct2 = tree.search_knn(&query, 2);
            let proxy_result2 = proxy.search_knn(&query, 2);
            assert_eq!(direct2, proxy_result2);
        }
    }
}
