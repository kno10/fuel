use std::cmp::Ordering;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::shared::brute_force_knn;
use super::super::VPTree;
use crate::{DataAccess, EuclideanDistance, MatrixDataAccess};

#[test]
fn test_search_knn_small_dataset() {
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

    let result = tree.search_knn(&dataset.with_query(&points[0]), 3);

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].index(), 0);

    let indices_1_2: Vec<usize> = result[1..3].iter().map(super::super::DistPair::index).collect();
    assert!(indices_1_2.contains(&1) && indices_1_2.contains(&2));

    let result = tree.search_knn(&dataset.with_query(&points[4]), 2);

    assert_eq!(result.len(), 2);
    assert_eq!(result[0].index(), 4);
    assert_eq!(result[1].index(), 3);
}

#[test]
fn test_edge_cases() {
    let points = vec![vec![0.0, 0.0]];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(42);

    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);
    assert_eq!(tree.points.len(), 1);

    let result = tree.search_knn(&dataset.with_query(&points[0]), 1);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].index(), 0);

    let result = tree.search_knn(&dataset, 0);
    assert_eq!(result.len(), 0);
}

#[test]
fn test_grid_search() {
    let mut points = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            points.push(vec![f64::from(x), f64::from(y)]);
        }
    }
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(42);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let center_idx = 2 * 5 + 2;
    let result = tree.search_knn(&dataset.with_query(&points[center_idx]), 5);

    assert_eq!(result.len(), 5);
    assert_eq!(result[0].index(), center_idx);
    assert!(result[0].distance() < 1e-10);

    let adjacent_indices = [7, 17, 11, 13];
    for neighbor in result.iter().take(5).skip(1) {
        assert!(adjacent_indices.contains(&neighbor.index()));
        assert!((neighbor.distance() - 1.0).abs() < 1e-10);
    }
}

#[test]
fn test_knn_against_brute_force() {
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
    let result = tree.search_knn(&dataset.with_query(&points[query_idx]), k);

    assert_eq!(result.len(), k);
    for i in 1..k {
        assert!(result[i - 1].distance() <= result[i].distance());
    }

    let mut distances: Vec<(f64, usize)> = (0..100)
        .map(|i| (dataset.distance(query_idx, i), i))
        .collect();
    distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let kth_distance = distances[k - 1].0;
    for neighbor in &result {
        assert!(neighbor.distance() <= kth_distance + 1e-10);
    }

    let strict_closer: Vec<usize> = distances
        .iter()
        .take_while(|(d, _)| *d < kth_distance - 1e-10)
        .map(|(_, idx)| *idx)
        .collect();

    for idx in strict_closer {
        assert!(result.iter().any(|dp| dp.index() == idx));
    }
}

#[test]
fn test_knn_matches_bruteforce_multiple_queries() {
    let mut points = Vec::new();
    for i in 0..60 {
        points.push(vec![
            f64::from(i) * 0.37,
            f64::from(i * 17 % 97) * 0.23 + f64::from(i * 7 % 13) * 0.001,
        ]);
    }

    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(123);
    let tree: VPTree<f64> = VPTree::new(&dataset, 4, rng);

    let query_indices = [0, 1, 7, 18, 33, 42, 59];
    let k_values = [1, 2, 3, 5, 10, 20];

    for &query_idx in &query_indices {
        for &k in &k_values {
            let knn = tree.search_knn(&dataset.with_query_index(query_idx), k);
            let brute = brute_force_knn(&dataset, query_idx, k);
            let mut all_distances: Vec<(f64, usize)> = dataset
                .iter()
                .map(|i| (dataset.distance(query_idx, i), i))
                .collect();
            all_distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });

            assert_eq!(knn.len(), brute.len());
            for i in 1..knn.len() {
                assert!(knn[i - 1].distance() <= knn[i].distance());
            }

            for (a, b) in knn.iter().zip(brute.iter()) {
                assert!((a.distance() - b.distance()).abs() < 1e-10);
            }

            let kth_distance = all_distances[k - 1].0;
            for neighbor in &knn {
                assert!(neighbor.distance() <= kth_distance + 1e-10);
            }

            let strict_closer: Vec<usize> = all_distances
                .iter()
                .take_while(|(d, _)| *d < kth_distance - 1e-10)
                .map(|(_, idx)| *idx)
                .collect();

            for idx in strict_closer {
                assert!(knn.iter().any(|dp| dp.index() == idx));
            }
        }
    }
}

#[test]
fn test_knn_with_k_larger_than_dataset() {
    let points = vec![
        vec![0.0, 0.0],
        vec![2.0, 0.0],
        vec![0.0, 2.0],
        vec![2.0, 2.0],
        vec![3.0, 1.0],
        vec![1.0, 3.0],
    ];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(7);
    let tree: VPTree<f64> = VPTree::new(&dataset, 2, rng);

    let query_idx = 3;
    let result = tree.search_knn(&dataset.with_query_index(query_idx), 100);
    let brute = brute_force_knn(&dataset, query_idx, dataset.size());

    assert_eq!(result.len(), dataset.size());
    for i in 1..result.len() {
        assert!(result[i - 1].distance() <= result[i].distance());
    }

    for (a, b) in result.iter().zip(brute.iter()) {
        assert!((a.distance() - b.distance()).abs() < 1e-10);
        assert_eq!(a.index(), b.index());
    }
}

#[test]
fn test_knn_self_is_nearest_for_all_queries() {
    let points = vec![
        vec![-1.0, 0.5],
        vec![0.2, -0.3],
        vec![1.8, 2.1],
        vec![-2.3, 1.4],
        vec![3.2, -1.9],
        vec![0.7, 3.5],
        vec![-1.6, -2.2],
    ];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(99);
    let tree: VPTree<f64> = VPTree::new(&dataset, 3, rng);

    for query_idx in 0..dataset.size() {
        let result = tree.search_knn(&dataset.with_query_index(query_idx), 4);
        assert!(!result.is_empty());
        assert_eq!(result[0].index(), query_idx);
        assert!(result[0].distance().abs() < 1e-12);
    }
}

#[test]
fn test_knn_random_fixed_seed_top5_matches_bruteforce() {
    let mut rng = StdRng::seed_from_u64(20_260_301);

    let mut points = Vec::with_capacity(200);
    for _ in 0..200 {
        points.push(vec![
            rng.gen_range(-500.0..500.0),
            rng.gen_range(-500.0..500.0),
        ]);
    }

    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let tree_rng = &mut StdRng::seed_from_u64(424_242);
    let tree: VPTree<f64> = VPTree::new(&dataset, 8, tree_rng);

    for _ in 0..20 {
        let query_idx = rng.gen_range(0..dataset.size());
        let knn = tree.search_knn(&dataset.with_query_index(query_idx), 5);

        assert_eq!(knn.len(), 5);
        for i in 1..knn.len() {
            assert!(knn[i - 1].distance() <= knn[i].distance());
        }

        let mut distances: Vec<(f64, usize)> = dataset
            .iter()
            .map(|i| (dataset.distance(query_idx, i), i))
            .collect();
        distances.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let brute_top5 = &distances[..5];
        let kth_distance = brute_top5[4].0;

        for (actual, (expected_dist, _)) in knn.iter().zip(brute_top5.iter()) {
            assert!((actual.distance() - *expected_dist).abs() < 1e-10);
        }

        for neighbor in &knn {
            assert!(neighbor.distance() <= kth_distance + 1e-10);
        }

        let strict_closer: Vec<usize> = distances
            .iter()
            .take_while(|(d, _)| *d < kth_distance - 1e-10)
            .map(|(_, idx)| *idx)
            .collect();

        for idx in strict_closer {
            assert!(knn.iter().any(|dp| dp.index() == idx));
        }
    }
}

#[test]
fn test_knn_with_external_query_data_matches_bruteforce() {
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
    let rng = &mut StdRng::seed_from_u64(20_260_301);
    let tree: VPTree<f64> = VPTree::new(&dataset, 2, rng);

    let k = 4;
    let knn = tree.search_knn(&dataset.with_query(&query), k);

    assert_eq!(knn.len(), k);
    for i in 1..knn.len() {
        assert!(knn[i - 1].distance() <= knn[i].distance());
    }

    let query_view = dataset.with_query(&query);
    let mut brute: Vec<(f64, usize)> = query_view
        .iter()
        .map(|i| (query_view.query_distance(i), i))
        .collect();
    brute.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let kth_distance = brute[k - 1].0;
    for (actual, (expected_dist, _)) in knn.iter().zip(brute.iter()) {
        assert!((actual.distance() - *expected_dist).abs() < 1e-10);
    }

    let strict_closer: Vec<usize> = brute
        .iter()
        .take_while(|(d, _)| *d < kth_distance - 1e-10)
        .map(|(_, idx)| *idx)
        .collect();

    for idx in strict_closer {
        assert!(knn.iter().any(|dp| dp.index() == idx));
    }
}
