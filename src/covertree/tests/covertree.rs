use std::collections::HashSet;

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::api::{Data, RangeSearch};
use crate::covertree::CoverTree;
use crate::distance::SquaredEuclidean;
use crate::{CoordinateQuery, DistPair, DistanceData, TableWithDistance};

fn sample_points() -> Vec<Vec<f64>> {
    vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]]
}

#[test]
fn cover_tree_knn_matches_brute() {
    let points = sample_points();
    let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
    let mut rng = StdRng::seed_from_u64(0xBEEFBEEF);
    let tree = CoverTree::new(&data, 1.3, 1, &mut rng);
    let query = data.query().with_coordinates(&points[0]);

    let neighbors: Vec<DistPair<f64>> = tree.search_knn(&query, 3);
    assert!(neighbors.len() >= 3);
    assert_eq!(neighbors[0].index, 0);

    let mut expected: Vec<DistPair<f64>> =
        (0..data.size()).map(|i| DistPair::new(data.distance(0, i), i)).collect();
    expected
        .sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    expected.truncate(3);

    let expected_ids: HashSet<usize> = expected.iter().map(|n| n.index).collect();
    let neighbor_ids: HashSet<usize> = neighbors.iter().map(|n| n.index).collect();
    assert_eq!(expected_ids, neighbor_ids);
}

#[test]
fn cover_tree_range_finds_close_points() {
    let points = sample_points();
    let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
    let mut rng = StdRng::seed_from_u64(0xBEEFBEEF);
    let tree = CoverTree::new(&data, 1.3, 1, &mut rng);
    let query = data.query().with_coordinates(&points[0]);

    let result: Vec<DistPair<f64>> = RangeSearch::search_range(&tree, &query, 1.01);

    assert!(result.iter().any(|n| n.index == 0));
    assert!(result.iter().any(|n| n.index == 1));
    assert!(result.iter().any(|n| n.index == 2));
    assert!(result.iter().all(|n| n.distance <= 1.01));
}

#[test]
fn cover_tree_priority_search_can_decrease_cutoff() {
    let points = sample_points();
    let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
    let mut rng = StdRng::seed_from_u64(0xBEEFBEEF);
    let tree = CoverTree::new(&data, 1.3, 1, &mut rng);
    let query = data.query().with_coordinates(&points[0]);
    let mut searcher = tree.priority_searcher();

    let first = searcher.next(&query).expect("should return first neighbor");
    assert_eq!(first.index, 0);

    searcher.decrease_cutoff(0.5);
    assert!(searcher.next(&query).is_none());
}

#[test]
fn cover_tree_priority_order_matches_knn() {
    let points = sample_points();
    let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
    let mut rng = StdRng::seed_from_u64(0xBEEFBEEF);
    let tree = CoverTree::new(&data, 1.3, 1, &mut rng);
    let query = data.query().with_coordinates(&points[0]);

    let knn: Vec<DistPair<f64>> = tree.search_knn(&query, 4);
    let mut searcher = tree.priority_searcher();
    let mut ks: Vec<DistPair<f64>> = Vec::new();
    for _ in 0..4 {
        if let Some(neighbor) = searcher.next(&query) {
            ks.push(neighbor);
        }
    }

    assert!(ks.len() >= 4);
    for (a, b) in ks.iter().zip(knn.iter()) {
        assert_eq!(a.index, b.index);
        let diff: f64 = (a.distance - b.distance).abs();
        assert!(diff < 1e-6);
    }
}
