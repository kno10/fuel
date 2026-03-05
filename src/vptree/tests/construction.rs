use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::VPTree;
use crate::{DataAccess, EuclideanDistance, MatrixDataAccess};

#[test]
fn test_vptree_construction() {
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

    assert_eq!(tree.points.len(), 5);
    assert_eq!(tree.bounds.len(), 5);

    let mut vp_indices = tree.points;
    vp_indices.sort_unstable();
    assert_eq!(vp_indices, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_vp_tree_with_sampling() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
        vec![4.0, 4.0],
        vec![5.0, 5.0],
    ];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(42);

    let tree: VPTree<f64> = VPTree::new(&dataset, 3, rng);

    assert_eq!(tree.points.len(), dataset.size());
    assert_eq!(tree.bounds.len(), dataset.size());

    let mut vp_indices = tree.points;
    vp_indices.sort_unstable();
    assert_eq!(vp_indices, vec![0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_sample_size_one_supports_all_searchers() {
    let points = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 1.0],
    ];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);
    let rng = &mut StdRng::seed_from_u64(1234);

    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let query_idx = 0;
    let knn = tree.search_knn(&dataset.with_query_index(query_idx), 3);
    assert_eq!(knn.len(), 3);

    let radius = knn.last().expect("kNN must be non-empty").distance();
    let mut range = Vec::new();
    tree.search_range(&dataset.with_query_index(query_idx), radius, |pair| {
        range.push(pair);
    });
    assert!(range.len() >= knn.len());

    let mut priority = tree.priority_searcher(dataset.with_query_index(query_idx));
    priority.set_threshold(radius);
    let priority_result = priority.get_all_neighbors();
    assert!(priority_result.len() >= knn.len());

    for p in &knn {
        assert!(range.iter().any(|r| r.index() == p.index()));
        assert!(priority_result.iter().any(|r| r.index() == p.index()));
    }
}

#[test]
fn test_matrix_data_access_supports_f32_points() {
    let points = vec![vec![0.0_f32, 0.0_f32], vec![3.0_f32, 4.0_f32]];
    let dataset = MatrixDataAccess::with_distance(&points, EuclideanDistance);

    assert!((dataset.distance(0, 1) - 5.0).abs() < 1e-6);
}
