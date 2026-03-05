use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{DataAccess, DistanceFunction, MatrixDataAccess, PrioritySearcher, VPTree};

use super::hdbscan_common::{
    HdbscanHierarchy, compute_core_distances, mutual_reachability_distance_from_distance,
};
use super::search_single_link_common::{ClusterBuilder, IndexedQueryData};

#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f64,
    point: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point && self.dist == other.dist
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.point.cmp(&self.point))
    }
}

/// Restarting-search HDBSCAN MST (HDBSCANRS port from ELKI).
#[must_use]
pub fn restarting_search_hdbscan<'t, 'm, 'd, T, DF>(
    tree: &'t VPTree,
    data: &'m MatrixDataAccess<'d, T, DF>,
    min_points: usize,
) -> HdbscanHierarchy
where
    DF: DistanceFunction<T>,
{
    let n = data.size();
    assert!(n > 0, "number of points must be positive");
    assert!(min_points > 0, "min_points must be greater than 0");

    let core_distances = compute_core_distances(data, min_points);
    if n == 1 {
        return HdbscanHierarchy::new(Vec::new(), core_distances);
    }

    let mut builder = ClusterBuilder::new(n);
    let mut heap = BinaryHeap::new();
    let mut best_dist = vec![f64::INFINITY; n];
    let mut neighbors = vec![None; n];
    let mut searcher = tree.priority_searcher(IndexedQueryData {
        data,
        query_index: 0,
    });
    let mut last_query = None;

    for a in 0..n {
        if builder.cluster_size_of_point(a) > 1 {
            continue;
        }
        if let Some((dist, b)) = find_next_neighbor(
            &mut searcher,
            data,
            &mut builder,
            &core_distances,
            a,
            0.0,
            &mut last_query,
        ) {
            best_dist[a] = dist;
            neighbors[a] = Some(b);
            heap.push(Candidate { dist, point: a });
        }
    }

    while builder.merge_count() < n - 1 {
        let Some(top) = heap.pop() else {
            break;
        };
        let a = top.point;
        let Some(b) = neighbors[a] else {
            continue;
        };
        if top.dist.to_bits() != best_dist[a].to_bits() {
            continue;
        }
        if !builder.same_set(a, b) {
            let _ = builder.merge_points(a, b, top.dist);
            if builder.merge_count() == n - 1 {
                break;
            }
        }

        let skip = if top.dist >= core_distances[a] {
            top.dist
        } else {
            0.0
        };
        if let Some((dist, next)) = find_next_neighbor(
            &mut searcher,
            data,
            &mut builder,
            &core_distances,
            a,
            skip,
            &mut last_query,
        ) {
            best_dist[a] = dist;
            neighbors[a] = Some(next);
            heap.push(Candidate { dist, point: a });
        } else {
            best_dist[a] = f64::INFINITY;
            neighbors[a] = None;
        }
    }

    HdbscanHierarchy::new(builder.into_history(), core_distances)
}

fn find_next_neighbor<'t, 'm, 'd, T, DF>(
    searcher: &mut PrioritySearcher<'t, IndexedQueryData<'m, 'd, T, DF>, f64>,
    data: &'m MatrixDataAccess<'d, T, DF>,
    builder: &mut ClusterBuilder,
    core_distances: &[f64],
    query_index: usize,
    skip: f64,
    last_query: &mut Option<usize>,
) -> Option<(f64, usize)>
where
    DF: DistanceFunction<T>,
{
    let skip = skip.max(0.0);
    if last_query.map_or(true, |last| last != query_index) {
        searcher.reset_with_data(IndexedQueryData { data, query_index });
        *last_query = Some(query_index);
    }
    searcher.increase_skip(skip);

    let cd = core_distances[query_index];
    let mut best_dist = f64::INFINITY;
    let mut best = None;

    while searcher.all_lower_bound() < best_dist {
        let Some(cand) = searcher.next() else {
            break;
        };
        let b = cand.index();
        if b == query_index || builder.same_set(query_index, b) {
            continue;
        }
        let d = cand.distance();
        if d < skip {
            continue;
        }
        if d == 0.0 {
            let _ = builder.merge_points(query_index, b, cd);
            continue;
        }
        let rd = mutual_reachability_distance_from_distance(core_distances, query_index, b, d);
        if rd < best_dist {
            best_dist = rd;
            best = Some(b);
            searcher.decrease_cutoff(best_dist);
        }
    }

    best.map(|idx| (best_dist, idx))
}

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::hdbscan_linear_memory;
    use crate::{EuclideanDistance, MatrixDataAccess, VPTree};
    use rand::{SeedableRng, rngs::StdRng};

    use super::restarting_search_hdbscan;

    #[test]
    fn restarting_search_hdbscan_matches_linear_mst() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![1.0, 1.2],
            vec![3.0, 3.0],
            vec![3.2, 3.1],
            vec![10.0, 10.0],
        ];
        let data = MatrixDataAccess::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(11);
        let tree = VPTree::new(&data, 3, &mut rng);

        let expected = hdbscan_linear_memory(&data, 2);
        let got = restarting_search_hdbscan(&tree, &data, 2);
        assert_eq!(got, expected);
    }
}
