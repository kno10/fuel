use std::cmp::Ordering;

use crate::DistanceData;
use num_traits::Float;

use crate::DistPair;
use crate::{DistanceSearch, vptree::PrioritySearcher};

/// Retrieve every neighbor produced by a priority searcher until exhaustion.
///
/// This duplicates the logic previously defined on the main type but is kept
/// inside the test hierarchy so that production code remains clean.
pub fn get_all_neighbors<F: Float, D: DistanceSearch<F>>(
    searcher: &mut PrioritySearcher<F>,
    data: &D,
) -> Vec<DistPair<F>> {
    let mut out = Vec::new();
    while let Some(n) = searcher.next_filtered(data, |_| false) {
        out.push(n);
    }
    out
}

pub fn brute_force_knn<T, S>(dataset: &T, query_idx: usize, k: usize) -> Vec<DistPair<S>>
where
    T: DistanceData<S>,
    S: Float + Copy + PartialOrd,
{
    let mut distances: Vec<(S, usize)> = dataset
        .iter()
        .map(|i| (dataset.distance(query_idx, i), i))
        .collect();

    distances.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    distances
        .into_iter()
        .take(k.min(dataset.size()))
        .map(|(distance, index)| DistPair::new(distance, index))
        .collect()
}
