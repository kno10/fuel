use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::search::covertree::CoverTree;
use crate::{DistPair, DistanceSearch, Float, KNNHeap};

#[derive(Debug, Clone, Copy)]
struct NodeEntry<F>
where
    F: Float,
{
    lower_bound: F,
    node_idx: u32,
    emit_center: bool,
    center_dist: F,
}

impl<F: Float> PartialEq for NodeEntry<F> {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound
            .partial_cmp(&other.lower_bound)
            .map(|o| o == Ordering::Equal)
            .unwrap_or(false)
    }
}

impl<F: Float> Eq for NodeEntry<F> {}

impl<F: Float> PartialOrd for NodeEntry<F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<F: Float> Ord for NodeEntry<F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower_bound.partial_cmp(&self.lower_bound).unwrap_or(Ordering::Equal)
    }
}

impl<F: Float> CoverTree<F> {
    pub fn search_knn<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, k: usize,
    ) -> Vec<DistPair<F>> {
        if k == 0 {
            return Vec::new();
        }

        let mut candidates: KNNHeap<F> = KNNHeap::new(k);
        let mut node_heap: BinaryHeap<NodeEntry<F>> = BinaryHeap::new();

        let root = &self.nodes[0];
        let root_center_dist = query.query_distance(root.center);
        let root_lower = root_center_dist - root.max_dist;
        node_heap.push(NodeEntry {
            lower_bound: root_lower,
            node_idx: 0,
            emit_center: true,
            center_dist: root_center_dist,
        });

        while let Some(entry) = node_heap.pop() {
            let lower = entry.lower_bound;
            let node = &self.nodes[entry.node_idx as usize];

            let current_tau = candidates.k_distance();
            if lower > current_tau {
                break;
            }

            let d_center = entry.center_dist;
            if entry.emit_center && d_center <= current_tau {
                candidates.insert(DistPair::new(d_center, node.center));
            }

            for singleton in node.singletons.iter() {
                let idx = singleton.index;
                let stored_dist = singleton.distance;
                if (d_center - stored_dist).abs() <= current_tau {
                    let d = query.query_distance(idx);
                    if d <= current_tau {
                        candidates.insert(DistPair::new(d, idx));
                    }
                }
            }

            let new_tau = candidates.k_distance();

            for &child_idx in node.children.iter() {
                let child = &self.nodes[child_idx as usize];

                // parent-child pruning bound: if child centroid cannot be within
                // radius new_tau even under best case, skip distance eval.
                let parent_child_bound = (d_center - child.parent_dist).abs();
                if parent_child_bound - child.max_dist > new_tau {
                    continue;
                }

                let d_child = query.query_distance(child.center);
                let child_lower = d_child - child.max_dist;
                if child_lower <= new_tau {
                    node_heap.push(NodeEntry {
                        lower_bound: child_lower,
                        node_idx: child_idx,
                        emit_center: child.center != node.center,
                        center_dist: d_child,
                    });
                }
            }
        }

        candidates.into_vec()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::data::Data;
    use crate::distance::SquaredEuclidean;
    use crate::search::covertree::CoverTree;
    use crate::{CoordinateQuery, DistPair, DistanceData, TableWithDistance};

    fn sample_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]]
    }

    #[test]
    fn cover_tree_knn_matches_brute() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = CoverTree::new(&data, 1.3, 0);
        let query = data.query().with_coordinates(&points[0]);

        let neighbors: Vec<DistPair<f64>> = tree.search_knn(&query, 3);
        assert!(neighbors.len() >= 3);
        assert_eq!(neighbors[0].index, 0);

        let mut expected: Vec<DistPair<f64>> =
            (0..data.len()).map(|i| DistPair::new(data.distance(0, i), i)).collect();
        expected.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        expected.truncate(3);

        let expected_ids: HashSet<usize> = expected.iter().map(|n| n.index).collect();
        let neighbor_ids: HashSet<usize> = neighbors.iter().map(|n| n.index).collect();
        assert_eq!(expected_ids, neighbor_ids);
    }
}
