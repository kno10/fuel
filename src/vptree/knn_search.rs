use std::{cmp::Ordering, cmp::Reverse, collections::BinaryHeap};

use num_traits::Float;

use crate::DataAccess;

use super::{DistPair, VPTree};

impl<F: Float> VPTree<F> {
    /// Find k nearest neighbors to the query point
    pub fn search_knn<D: DataAccess>(&self, data: &D, k: usize) -> Vec<DistPair<F>> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<Reverse<DistPair<F>>> = BinaryHeap::with_capacity(k + 1);
        self.search_knn_recursive(data, k, 0, self.points.len(), &mut heap);

        let mut result: Vec<DistPair<F>> = heap.into_iter().map(|item| item.0).collect();
        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        result
    }

    /// Recursively search for k nearest neighbors
    fn search_knn_recursive<D: DataAccess>(
        &self,
        data: &D,
        k: usize,
        left: usize,
        right: usize,
        heap: &mut BinaryHeap<Reverse<DistPair<F>>>,
    ) -> F {
        let node_idx = left;
        let vp = self.points[node_idx];

        // Distance to vantage point
        let d = F::from(data.query_distance(vp as usize))
            .expect("distance cannot be represented by target float type");

        // Add vantage point to candidates
        if heap.len() < k {
            heap.push(Reverse(DistPair::new(d, vp as usize)));
        } else if d < heap.peek().unwrap().0.distance {
            heap.pop();
            heap.push(Reverse(DistPair::new(d, vp as usize)));
        }

        // Current tau (distance to k-th nearest neighbor)
        let mut tau = if heap.len() < k {
            F::infinity()
        } else {
            heap.peek().unwrap().0.distance
        };

        if left + 1 >= right {
            return tau;
        }

        let mid = usize::midpoint(left, right);

        let left_child = if left + 1 < mid {
            let child_left = left + 1;
            let child = self.bounds[child_left];
            let min_dist = if d < child.lower {
                child.lower - d
            } else if d > child.upper {
                d - child.upper
            } else {
                F::zero()
            };
            Some((child_left, mid, min_dist))
        } else {
            None
        };

        let right_child = if mid < right {
            let child = self.bounds[mid];
            let min_dist = if d < child.lower {
                child.lower - d
            } else if d > child.upper {
                d - child.upper
            } else {
                F::zero()
            };
            Some((mid, right, min_dist))
        } else {
            None
        };

        match (left_child, right_child) {
            (Some(left_node), Some(right_node)) => {
                let (first, second) = if left_node.2 <= right_node.2 {
                    (left_node, right_node)
                } else {
                    (right_node, left_node)
                };

                if first.2 <= tau {
                    tau = self.search_knn_recursive(data, k, first.0, first.1, heap);
                }
                if second.2 <= tau {
                    self.search_knn_recursive(data, k, second.0, second.1, heap);
                }
            }
            (Some(node), None) | (None, Some(node)) if node.2 <= tau => {
                self.search_knn_recursive(data, k, node.0, node.1, heap);
            }
            _ => {}
        }

        if heap.len() < k {
            F::infinity()
        } else {
            heap.peek().unwrap().0.distance
        }
    }
}
