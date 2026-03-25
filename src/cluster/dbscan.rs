use std::collections::HashSet;

use crate::api::RangeSearch;
use crate::{DistanceData, Float, IndexQuery};

const UNVISITED: isize = -2;
pub const NOISE: isize = -1;

/// Run DBSCAN using a generic range search index for neighborhood queries.
///
/// Returns a label per point:
/// - `NOISE` (-1) for noise points
/// - `0..` for cluster ids
pub fn dbscan<'a, S, D, F>(tree: &S, data: &'a D, eps: F, min_points: usize) -> Vec<isize>
where
    F: Float,
    D: DistanceData<F> + 'a,
    S: RangeSearch<F, D::Query<'a>>,
{
    assert!(eps >= F::zero(), "eps must be non-negative");

    let size = data.size();
    let mut labels = vec![UNVISITED; size];
    let mut cluster_id: isize = 0;
    let mut frontier = HashSet::new();
    let mut neighbors = Vec::with_capacity(min_points);

    let mut query = data.query();
    for point_idx in 0..size {
        if labels[point_idx] != UNVISITED {
            continue;
        }
        frontier.clear();

        query.set_index(point_idx);
        for pair in tree.search_range(&query, eps) {
            frontier.insert(pair.index);
        }
        if frontier.len() < min_points {
            labels[point_idx] = NOISE;

            continue;
        }

        // Start new cluster
        labels[point_idx] = cluster_id;

        while let Some(&current_idx) = frontier.iter().next() {
            frontier.remove(&current_idx);

            if labels[current_idx] == NOISE {
                labels[current_idx] = cluster_id;
            }
            if labels[current_idx] != UNVISITED {
                continue;
            }
            labels[current_idx] = cluster_id;

            neighbors.clear();
            query.set_index(current_idx);
            for pair in tree.search_range(&query, eps) {
                neighbors.push(pair.index);
            }
            if neighbors.len() >= min_points {
                for &neighbor_idx in &neighbors {
                    if labels[neighbor_idx] == NOISE {
                        labels[neighbor_idx] = cluster_id;
                    }
                    if labels[neighbor_idx] == UNVISITED {
                        frontier.insert(neighbor_idx);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    labels
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::EuclideanDistance;
    use crate::vptree::VPTree;

    #[test]
    fn dbscan_finds_two_clusters_and_noise() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
            vec![5.0, 5.0],
        ];

        let data = TableWithDistance::with_distance(&points, EuclideanDistance);
        let mut rng = StdRng::seed_from_u64(7);
        let tree = VPTree::new(&data, 2, &mut rng);

        let labels = dbscan(&tree, &data, 0.25, 3);

        assert_eq!(labels.len(), points.len());
        assert_eq!(labels[6], NOISE);

        let first_cluster = labels[0];
        let second_cluster = labels[3];
        assert!(first_cluster >= 0);
        assert!(second_cluster >= 0);
        assert_ne!(first_cluster, second_cluster);

        assert_eq!(labels[1], first_cluster);
        assert_eq!(labels[2], first_cluster);
        assert_eq!(labels[4], second_cluster);
        assert_eq!(labels[5], second_cluster);

        let clusters: HashSet<isize> = labels.iter().copied().filter(|&label| label >= 0).collect();
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn dbscan_matches_sklearn_toy_labels() {
        let points =
            vec![vec![0.0], vec![2.0], vec![3.0], vec![4.0], vec![6.0], vec![8.0], vec![10.0]];

        let expected_cases = [
            (1, vec![0, 1, 1, 1, 2, 3, 4]),
            (2, vec![NOISE, 0, 0, 0, NOISE, NOISE, NOISE]),
            (3, vec![NOISE, 0, 0, 0, NOISE, NOISE, NOISE]),
            (4, vec![NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, NOISE]),
        ];

        for (min_points, expected_labels) in expected_cases {
            let data = TableWithDistance::with_distance(&points, EuclideanDistance);
            let mut rng = StdRng::seed_from_u64(7);
            let tree = VPTree::new(&data, 2, &mut rng);

            let labels = dbscan(&tree, &data, 1.0, min_points);
            assert_eq!(labels, expected_labels);
        }
    }
}
