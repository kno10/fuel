use std::cmp::Ordering;

use crate::DataAccess;

use super::super::DistPair;

pub fn brute_force_knn<T: DataAccess>(dataset: &T, query_idx: usize, k: usize) -> Vec<DistPair> {
    let mut distances: Vec<(f64, usize)> = dataset
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
