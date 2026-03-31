use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::search::covertree::CoverTree;
use crate::{ApproxKnnSearch, DistPair, DistanceSearch, Float, KNNHeap};

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
    pub fn search_aknn<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, k: usize, rate: f32,
    ) -> Vec<DistPair<F>> {
        if k == 0 || self.nodes.is_empty() || rate <= 0.0 || !rate.is_finite() {
            return Vec::new();
        }

        if rate >= 1.0 {
            return self.search_knn(query, k);
        }

        let max_dists = (rate * (self.nodes.len() as f32)).ceil() as usize;
        let max_dists = max_dists.min(self.nodes.len());
        if max_dists == 0 {
            return Vec::new();
        }

        let mut candidates: KNNHeap<F> = KNNHeap::new(k);
        let mut node_heap: BinaryHeap<NodeEntry<F>> = BinaryHeap::new();
        let mut dist_count = 0_usize;

        if dist_count < max_dists {
            let root_idx = 0;
            let root = &self.nodes[root_idx as usize];
            let root_center_dist = query.query_distance(root.center);
            dist_count += 1;
            let root_lower = root_center_dist - root.max_dist;
            node_heap.push(NodeEntry {
                lower_bound: root_lower,
                node_idx: root_idx,
                emit_center: true,
                center_dist: root_center_dist,
            });
        }

        while let Some(entry) = node_heap.pop() {
            if dist_count > max_dists {
                break;
            }

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

            if dist_count >= max_dists {
                break;
            }

            for singleton in node.singletons.iter() {
                if dist_count >= max_dists {
                    break;
                }
                let idx = singleton.index;
                let stored_dist = singleton.distance;
                if (d_center - stored_dist).abs() <= current_tau {
                    if dist_count >= max_dists {
                        break;
                    }
                    let d = query.query_distance(idx);
                    dist_count += 1;
                    if d <= current_tau {
                        candidates.insert(DistPair::new(d, idx));
                    }
                }
            }

            let new_tau = candidates.k_distance();

            for &child_idx in node.children.iter() {
                if dist_count >= max_dists {
                    break;
                }
                let child = &self.nodes[child_idx as usize];

                let parent_child_bound = (d_center - child.parent_dist).abs();
                if parent_child_bound - child.max_dist > new_tau {
                    continue;
                }

                let d_child = query.query_distance(child.center);
                dist_count += 1;
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

impl<F: Float, Q: DistanceSearch<F> + ?Sized> ApproxKnnSearch<F, Q> for CoverTree<F> {
    fn search_aknn(&self, query: &Q, k: usize, rate: f32) -> Vec<DistPair<F>> {
        self.search_aknn(query, k, rate)
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::SquaredEuclidean;
    use crate::search::covertree::CoverTree;
    use crate::{CoordinateQuery, DistPair, DistanceData, TableWithDistance};

    fn sample_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]]
    }

    #[test]
    fn covertree_aknn_budget_behaves_reasonably() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = CoverTree::new(&data, 1.3, 0);
        let query = data.query().with_coordinates(&points[0]);

        let exact: Vec<DistPair<f64>> = tree.search_knn(&query, 3);
        let approx_full: Vec<DistPair<f64>> = tree.search_aknn(&query, 3, 1.0);
        assert_eq!(exact, approx_full);

        let approx_small: Vec<DistPair<f64>> = tree.search_aknn(&query, 3, 0.2);
        assert!(!approx_small.is_empty());
        assert!(approx_small.len() <= 3);

        let zero: Vec<DistPair<f64>> = tree.search_aknn(&query, 3, 0.0);
        assert!(zero.is_empty());
    }
}
