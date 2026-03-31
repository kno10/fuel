use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::VPTree;
use crate::api::DistanceData;
use crate::distance::Euclidean;
use crate::{CoordinateQuery, DistanceSearch, IndexQuery, TableWithDistance};

#[test]
fn test_range_search() {
    let mut points = Vec::new();
    for y in 0..5 {
        for x in 0..5 {
            points.push(vec![f64::from(x), f64::from(y)]);
        }
    }
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(42);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let center_idx = 2 * 5 + 2;
    let mut result = Vec::new();
    let query = dataset.query().with_coordinates(&points[center_idx]);
    tree.search_range(&query, 1.5, |pair| {
        result.push(pair);
    });

    assert_eq!(result.len(), 9);
    for dist_pair in &result {
        assert!(dist_pair.distance <= 1.5);
    }

    assert_eq!(result[0].index, center_idx);
    assert!(result[0].distance < 1e-10);
}

#[test]
fn test_range_search_zero_radius_returns_self_only() {
    let points = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(9001);
    let tree: VPTree<f64> = VPTree::new(&dataset, 2, rng);

    let mut result = Vec::new();
    let query = dataset.query().with_index(2);
    tree.search_range(&query, 0.0, |pair| {
        result.push(pair);
    });
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].index, 2);
    assert!(result[0].distance.abs() < 1e-12_f64);
}

#[test]
fn test_query_can_be_external_slice() {
    let points = vec![vec![0.0, 0.0], vec![3.0, 4.0], vec![1.0, 1.0]];
    let query = vec![0.5, 0.5];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);

    let expected0 = 0.5f64.hypot(0.5f64);
    let expected1 = 2.5f64.hypot(3.5f64);

    let query_view = dataset.query().with_coordinates(&query);
    assert!((DistanceSearch::<f64>::query_distance(&query_view, 0) - expected0).abs() < 1e-12);
    assert!((DistanceSearch::<f64>::query_distance(&query_view, 1) - expected1).abs() < 1e-12);
}
