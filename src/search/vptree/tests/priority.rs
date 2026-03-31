use std::cell::Cell;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::super::VPTree;
use super::shared::get_all_neighbors;
use crate::api::{Data, DistanceData, DistanceSearch};
use crate::distance::Euclidean;
use crate::search::vptree::{NodePoints, SearchFilter};
use crate::{CoordinateQuery, Float, IndexQuery, TableQuery, TableWithDistance};

#[derive(Clone, Copy)]
struct CountingQueryData<'a, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: crate::distance::DistanceFunction<[C], F>,
    F: Float,
{
    // we will implement DistanceSearch for this wrapper as well
    data: &'a TableWithDistance<'a, C, T, DF, F>,
    query_index: usize,
    query_calls: &'a Cell<usize>,
}

// `DistanceData` extends `Data`, so implement `Data` separately
impl<'a, C, T, DF, F> crate::api::Data for CountingQueryData<'a, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: crate::distance::DistanceFunction<[C], F>,
    F: Float,
{
    fn len(&self) -> usize { self.data.len() }
}

impl<'a, C, T, DF, F> DistanceData<F> for CountingQueryData<'a, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: crate::distance::DistanceFunction<[C], F>,
    F: Float,
{
    type Query<'b>
        = TableQuery<'b, 'a, C, T, DF, F>
    where
        Self: 'b;

    fn distance(&self, a: usize, b: usize) -> F { self.data.distance(a, b) }

    fn query(&self) -> Self::Query<'_> { self.data.query() }
}

// implement DistanceSearch so the wrapper itself can be used as a query
impl<'a, C, T, DF, F> crate::api::DistanceSearch<F> for CountingQueryData<'a, C, T, DF, F>
where
    C: Float,
    T: AsRef<[C]>,
    DF: crate::distance::DistanceFunction<[C], F>,
    F: Float,
{
    fn query_distance(&self, b: usize) -> F {
        // increment counter and delegate
        self.query_calls.set(self.query_calls.get() + 1);
        self.data.distance(self.query_index, b)
    }
}

#[test]
fn test_priority_search() {
    let points =
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(42);

    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let mut searcher = tree.priority_searcher();

    let mut result = Vec::new();
    for _ in 0..3 {
        let query = dataset.query().with_coordinates(&points[0]);
        if let Some(neighbor) = searcher.next_filtered(&query, |_| false) {
            result.push(neighbor);
        }
    }

    assert_eq!(result.len(), 3);
    assert_eq!(result[0].index, 0);

    let indices_1_2: Vec<usize> = result[1..3].iter().map(|dp| dp.index).collect();
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
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(12345);
    let tree: VPTree<f64> = VPTree::new(&dataset, 3, rng);

    let mut reusable = tree.priority_searcher();
    reusable.decrease_cutoff(2.5);
    let query = dataset.query().with_index(0);
    let first = get_all_neighbors(&mut reusable, &query);

    let mut fresh = tree.priority_searcher();
    fresh.decrease_cutoff(2.5);
    let query = dataset.query().with_index(0);
    let first_fresh = get_all_neighbors(&mut fresh, &query);
    assert_eq!(first, first_fresh);

    reusable.reset();
    reusable.decrease_cutoff(3.0);
    reusable.increase_skip(0.5);
    let query = dataset.query().with_index(4);
    let second = get_all_neighbors(&mut reusable, &query);

    let mut fresh_second = tree.priority_searcher();
    fresh_second.decrease_cutoff(3.0);
    fresh_second.increase_skip(0.5);
    let query = dataset.query().with_index(4);
    let second_fresh = get_all_neighbors(&mut fresh_second, &query);
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

    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(314_159);
    let tree: VPTree<f64> = VPTree::new(&dataset, 3, rng);
    let query_idx = 2 * 5 + 2;
    let mut cutoff_searcher = tree.priority_searcher();
    cutoff_searcher.decrease_cutoff(1.5);
    let query = dataset.query().with_index(query_idx);
    let cutoff_result = get_all_neighbors(&mut cutoff_searcher, &query);

    assert!(!cutoff_result.is_empty());
    for p in &cutoff_result {
        assert!(p.distance <= 1.5 + 1e-12);
    }

    let mut skip_searcher = tree.priority_searcher();
    skip_searcher.decrease_cutoff(2.0);
    skip_searcher.increase_skip(1.0 + 1e-12);
    let query = dataset.query().with_index(query_idx);
    let skipped_result = get_all_neighbors(&mut skip_searcher, &query);

    assert!(!skipped_result.is_empty());
    for p in &skipped_result {
        assert!(p.distance >= 1.0 + 1e-12);
        assert!(p.distance <= 2.0 + 1e-12);
    }
    assert!(!skipped_result.iter().any(|p| p.index == query_idx));
}

#[test]
fn test_compare_search_methods() {
    let mut points = Vec::new();
    for y in 0..10 {
        for x in 0..10 {
            points.push(vec![f64::from(x), f64::from(y)]);
        }
    }
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(42);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let query_idx = 45;
    let k = 10;
    let query = dataset.query().with_index(query_idx);
    let knn_result = tree.search_knn(&query, k);

    let mut priority_searcher = tree.priority_searcher();
    let radius = knn_result.last().unwrap().distance;
    priority_searcher.decrease_cutoff(radius);
    let query = dataset.query().with_index(query_idx);
    let priority_result = get_all_neighbors(&mut priority_searcher, &query);
    let mut range_result = Vec::new();
    tree.search_range(&query, radius, |pair| {
        range_result.push(pair);
    });

    assert!(knn_result.len() >= k);
    assert!(priority_result.len() >= k);
    assert!(range_result.len() >= k);

    for knn in &knn_result {
        assert!(priority_result.iter().any(|dp| dp.index == knn.index));
        assert!(range_result.iter().any(|dp| dp.index == knn.index));
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
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(271_828);
    let tree: VPTree<f64> = VPTree::new(&dataset, 2, rng);

    let mut searcher = tree.priority_searcher();
    let query_view = dataset.query().with_coordinates(&query);
    let priority_all = get_all_neighbors(&mut searcher, &query_view);

    let query_view = dataset.query().with_coordinates(&query);
    let mut expected: Vec<(f64, usize)> =
        dataset.iter().map(|i| (query_view.query_distance(i), i)).collect();
    expected.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.1.cmp(&b.1))
    });

    assert_eq!(priority_all.len(), expected.len());

    let mut actual_distances: Vec<f64> = priority_all.iter().map(|dp| dp.distance).collect();
    actual_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for (actual, (expected_dist, _)) in actual_distances.iter().zip(expected.iter()) {
        assert!((*actual - *expected_dist).abs() < 1e-10);
    }
}

#[test]
fn test_priority_search_uses_bounds_to_prune_distance_computations() {
    let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![100.0], vec![101.0], vec![102.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(7);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let full_counter = Cell::new(0);
    let full_query =
        CountingQueryData { data: &dataset, query_index: 0, query_calls: &full_counter };
    let mut full_search = tree.priority_searcher();
    let dataset_size = dataset.len();
    let all_neighbors = get_all_neighbors(&mut full_search, &full_query);
    assert_eq!(all_neighbors.len(), dataset_size);
    assert_eq!(
        full_counter.get(),
        dataset_size,
        "unbounded search should evaluate each point exactly once"
    );

    let pruned_counter = Cell::new(0);
    let pruned_query =
        CountingQueryData { data: &dataset, query_index: 0, query_calls: &pruned_counter };
    let mut pruned_search = tree.priority_searcher();
    pruned_search.decrease_cutoff(2.5);
    let mut pruned_neighbors = get_all_neighbors(&mut pruned_search, &pruned_query);
    pruned_neighbors.sort_by_key(|dp| dp.index);

    assert_eq!(pruned_neighbors.len(), 3);
    assert_eq!(pruned_neighbors.iter().map(|dp| dp.index).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(
        pruned_counter.get(),
        3,
        "cutoff should prune the far subtree before computing distances"
    );
}

#[test]
fn test_priority_search_all_lower_bound_tracks_remaining_candidates() {
    let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![10.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(99);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let mut searcher = tree.priority_searcher();
    assert_eq!(searcher.all_lower_bound(), 0.0);

    let first = searcher
        .next_filtered(&dataset.query().with_index(0), |_| false)
        .expect("first candidate must exist");
    assert_eq!(first.index, 0);
    assert_eq!(first.distance, 0.0);

    assert!(
        searcher.all_lower_bound() > 0.0,
        "after consuming the self hit, lower bound must advance to the next node bound"
    );
}

#[test]
fn test_priority_search_uses_upper_bounds_for_skip_pruning() {
    let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![100.0], vec![101.0], vec![102.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(7);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let counter = Cell::new(0);
    let query = CountingQueryData { data: &dataset, query_index: 0, query_calls: &counter };
    let mut searcher = tree.priority_searcher();
    searcher.increase_skip(50.0);
    let mut neighbors = get_all_neighbors(&mut searcher, &query);
    neighbors.sort_by_key(|dp| dp.index);

    assert_eq!(neighbors.iter().map(|dp| dp.index).collect::<Vec<_>>(), vec![3, 4, 5]);
    assert_eq!(
        counter.get(),
        4,
        "skip should prune the near subtree and only visit root + far subtree"
    );
}

#[test]
fn test_priority_search_filter_skips_distance_computation() {
    let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(1234);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let counter = Cell::new(0);
    let query = CountingQueryData { data: &dataset, query_index: 0, query_calls: &counter };
    let mut searcher = tree.priority_searcher();

    let mut seen = Vec::new();
    while let Some(nei) = searcher.next_filtered(&query, |idx| idx == 2) {
        seen.push(nei.index);
    }
    assert!(!seen.contains(&2));
    assert!(counter.get() < points.len(), "skipped index should not require a query_distance call");
}

#[test]
fn test_priority_search_filter_skips_distance_computation_with_skip_threshold() {
    let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(1234);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let counter = Cell::new(0);
    let query = CountingQueryData { data: &dataset, query_index: 0, query_calls: &counter };
    let mut searcher = tree.priority_searcher();
    searcher.increase_skip(1.5);

    let mut seen = Vec::new();
    while let Some(nei) = searcher.next_filtered(&query, |idx| idx == 2) {
        seen.push(nei.index);
    }
    assert!(!seen.contains(&2));
    assert!(
        counter.get() < points.len(),
        "skipped index should not require a query_distance call even with skip threshold"
    );
}

struct SkipSubtreeFilter {
    skipped_points: Vec<usize>,
}

impl SearchFilter for SkipSubtreeFilter {
    fn skip_node(&mut self, points: NodePoints<'_>) -> bool {
        let indices: Vec<_> = points.indices().collect();
        if indices.len() >= 2 && indices.iter().all(|&index| (3..=5).contains(&index)) {
            self.skipped_points.extend(indices);
            return true;
        }
        false
    }

    fn skip_point(&mut self, index: usize) -> bool {
        self.skipped_points.push(index);
        false
    }
}

#[test]
fn test_priority_search_node_filter_skips_subtree_distance_computations() {
    let points = vec![vec![0.0], vec![1.0], vec![2.0], vec![100.0], vec![101.0], vec![102.0]];
    let dataset = TableWithDistance::with_distance(&points, Euclidean);
    let rng = &mut StdRng::seed_from_u64(7);
    let tree: VPTree<f64> = VPTree::new(&dataset, 1, rng);

    let counter = Cell::new(0);
    let query = CountingQueryData { data: &dataset, query_index: 0, query_calls: &counter };
    let mut searcher = tree.priority_searcher();
    let mut filter = SkipSubtreeFilter { skipped_points: Vec::new() };

    let neighbors: Vec<_> =
        std::iter::from_fn(|| searcher.next_with_filter(&query, &mut filter)).collect();

    assert_eq!(neighbors.iter().map(|dp| dp.index).collect::<Vec<_>>(), vec![0, 1, 2]);
    assert_eq!(
        counter.get(),
        3,
        "node filter should avoid query_distance calls for the skipped subtree"
    );
}
