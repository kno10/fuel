use crate::vptree::VPTree;
use crate::{ApproxKnnSearch, DistPair, DistanceSearch, Float, KNNHeap};

impl<F: Float> VPTree<F> {
    /// Find k nearest neighbors (approximate with a distance budget rate).
    pub fn search_aknn<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, k: usize, rate: f32,
    ) -> Vec<DistPair<F>> {
        if k == 0 || self.points.is_empty() || rate <= 0.0 || !rate.is_finite() {
            return Vec::new();
        }

        let max_dists = (rate * (self.points.len() as f32)).ceil() as usize;
        if max_dists == 0 {
            return Vec::new();
        }

        let max_dists = max_dists.min(self.points.len());
        let mut heap: KNNHeap<F> = KNNHeap::new(k);
        let mut dist_count = 0_usize;

        self.search_aknn_recursive(
            query,
            0,
            self.points.len(),
            &mut heap,
            &mut dist_count,
            max_dists,
        );

        heap.into_vec()
    }

    fn search_aknn_recursive<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, left: usize, right: usize, heap: &mut KNNHeap<F>, dist_count: &mut usize,
        max_dists: usize,
    ) -> (F, bool) {
        if left >= right || *dist_count >= max_dists {
            return (heap.k_distance(), *dist_count >= max_dists);
        }

        let node_idx = left;
        let vp = self.points[node_idx];

        let d = query.query_distance(vp as usize);
        *dist_count += 1;

        let mut tau = heap.insert(DistPair::new(d, vp as usize));
        if left + 1 >= right || *dist_count >= max_dists {
            return (tau, *dist_count >= max_dists);
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

        let mut stop = false;
        match (left_child, right_child) {
            (Some(left_node), Some(right_node)) => {
                let (first, second) = if left_node.2 <= right_node.2 {
                    (left_node, right_node)
                } else {
                    (right_node, left_node)
                };

                if first.2 <= tau {
                    let (new_tau, done) = self.search_aknn_recursive(
                        query, first.0, first.1, heap, dist_count, max_dists,
                    );
                    tau = new_tau;
                    if done {
                        return (tau, true);
                    }
                }

                if second.2 <= tau {
                    let (new_tau, done) = self.search_aknn_recursive(
                        query, second.0, second.1, heap, dist_count, max_dists,
                    );
                    tau = new_tau;
                    if done {
                        return (tau, true);
                    }
                }
            }
            (Some(node), None) | (None, Some(node)) if node.2 <= tau => {
                let (new_tau, done) =
                    self.search_aknn_recursive(query, node.0, node.1, heap, dist_count, max_dists);
                tau = new_tau;
                stop = done;
            }
            _ => {}
        }

        (tau, stop)
    }
}

impl<F: Float, Q: DistanceSearch<F> + ?Sized> ApproxKnnSearch<F, Q> for VPTree<F> {
    fn search_aknn(&self, query: &Q, k: usize, rate: f32) -> Vec<DistPair<F>> {
        self.search_aknn(query, k, rate)
    }
}
