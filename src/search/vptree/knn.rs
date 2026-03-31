use crate::search::vptree::VPTree;
use crate::{DistPair, DistanceSearch, Float, KNNHeap};

impl<F: Float> VPTree<F> {
    /// Find k nearest neighbors to the query point
    pub fn search_knn<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, k: usize,
    ) -> Vec<DistPair<F>> {
        if k == 0 {
            return Vec::new();
        }

        let mut heap: KNNHeap<F> = KNNHeap::new(k);
        self.search_knn_recursive(query, 0, self.points.len(), &mut heap);

        heap.into_vec()
    }

    /// Recursively search for k nearest neighbors
    fn search_knn_recursive<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, left: usize, right: usize, heap: &mut KNNHeap<F>,
    ) -> F {
        let node_idx = left;
        let vp = self.points[node_idx];

        // Distance to vantage point
        let d = query.query_distance(vp as usize);

        // Add vantage point to candidates
        let mut tau = heap.insert(DistPair::new(d, vp as usize));

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
                    tau = self.search_knn_recursive(query, first.0, first.1, heap);
                }
                if second.2 <= tau {
                    self.search_knn_recursive(query, second.0, second.1, heap);
                }
            }
            (Some(node), None) | (None, Some(node)) if node.2 <= tau => {
                self.search_knn_recursive(query, node.0, node.1, heap);
            }
            _ => {}
        }

        heap.k_distance()
    }
}
