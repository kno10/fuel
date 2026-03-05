use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::VPTree;
use crate::{DataAccess, EuclideanDistance, MatrixDataAccess};

#[test]
fn test_priority_search() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
    ];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(42);

    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let mut searcher = tree.priority_searcher(dataset.with_query(&points[0]));

    let mut result = Vec::new();
    for _ in 0..3 {
        if let Some(neighbor) = searcher.next() {
            result.push(neighbor);
        }
    }

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].index(), 0);

    let indices_1_2: Vec<usize> = result[1..3].iter().map(super::super::DistPair::index).collect();
    assert!(indices_1_2.contains(&1) && indices_1_2.contains(&2));
}

#[test]
fn test_priority_searcher_reuse_reset() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 0.5],
        vec![-1.0, 1.5],
    ];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(12345);
    let tree: VPTree<f64> = VPTree::new(&dataset, 3, rng);

    let mut reusable = tree.priority_searcher(dataset.with_query_index(0));
    reusable.decrease_cutoff(2.5);
    let first = reusable.get_all_neighbors();

    let mut fresh = tree.priority_searcher(dataset.with_query_index(0));
    fresh.decrease_cutoff(2.5);
    let first_fresh = fresh.get_all_neighbors();
    assert_eq!(first, first_fresh);

    reusable.reset_with_data(dataset.with_query_index(4));
    reusable.reset_with_limits(3.0, 0.5);
    let second = reusable.get_all_neighbors();

    let mut fresh_second = tree.priority_searcher(dataset.with_query_index(4));
    fresh_second.decrease_cutoff(3.0);
    fresh_second.increase_skip(0.5);
    let second_fresh = fresh_second.get_all_neighbors();
    assert_eq!(second, second_fresh);
}

#[test]
fn test_priority_search_cutoff_and_skip() {
    let mut points = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            points.push(vec![f64::from(x), f64::from(y)]);
        }
    }

    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(314_159);
    let tree: VPTree<f64> = VPTree::new(&dataset, 3, rng);
    let query_idx = 2 * 5 + 2;
    let mut cutoff_searcher = tree.priority_searcher(dataset.with_query_index(query_idx));
    cutoff_searcher.decrease_cutoff(1.5);
    let cutoff_result = cutoff_searcher.get_all_neighbors();

    assert!(!cutoff_result.is_empty());
    for p in &cutoff_result {
        assert!(p.distance() <= 1.5 + 1e-12);
    }

    let mut skip_searcher = tree.priority_searcher(dataset.with_query_index(query_idx));
    skip_searcher.decrease_cutoff(2.0);
    skip_searcher.increase_skip(1.0 + 1e-12);
    let skipped_result = skip_searcher.get_all_neighbors();

    assert!(!skipped_result.is_empty());
    for p in &skipped_result {
        assert!(p.distance() >= 1.0 + 1e-12);
        assert!(p.distance() <= 2.0 + 1e-12);
    }
    assert!(!skipped_result.iter().any(|p| p.index() == query_idx));
}

#[test]
fn test_compare_search_methods() {
    let mut points = Vec::new();
    for y in 0..10 {
        for x in 0..10 {
            points.push(vec![f64::from(x), f64::from(y)]);
        }
    }
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(42);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let query_idx = 45;
    let k = 10;
    let knn_result = tree.search_knn(&dataset.with_query(&points[query_idx]), k);

    let mut priority_searcher = tree.priority_searcher(dataset.with_query(&points[query_idx]));
    let radius = knn_result.last().unwrap().distance();
    priority_searcher.set_threshold(radius);
    let priority_result = priority_searcher.get_all_neighbors();
    let mut range_result = Vec::new();
    tree.search_range(&dataset.with_query(&points[query_idx]), radius, |pair| {
        range_result.push(pair);
    });

    assert_eq!(knn_result.len(), k);
    assert!(priority_result.len() >= k);
    assert!(range_result.len() >= k);

    for knn in &knn_result {
        assert!(priority_result.iter().any(|dp| dp.index() == knn.index()));
        assert!(range_result.iter().any(|dp| dp.index() == knn.index()));
    }
}

#[test]
fn test_priority_search_with_external_query_data() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 1.0],
    ];
    let query = vec![0.5, 0.25];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(271_828);
    let tree: VPTree<f64> = VPTree::new(&dataset, 2, rng);

    let mut searcher = tree.priority_searcher(dataset.with_query(&query));
    let priority_all = searcher.get_all_neighbors();

    let query_view = dataset.with_query(&query);
    let mut expected: Vec<(f64, usize)> = query_view
        .iter()
        .map(|i| (query_view.query_distance(i), i))
        .collect();
    expected.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    assert_eq!(priority_all.len(), expected.len());

    let mut actual_distances: Vec<f64> = priority_all.iter().map(super::super::DistPair::distance).collect();
    actual_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for (actual, (expected_dist, _)) in actual_distances.iter().zip(expected.iter()) {
        assert!((*actual - *expected_dist).abs() < 1e-10);
    }
}
