//! Cover tree kNN search with pruning on node lower bounds.
//!
//! Key invariants:
//! - `node.max_dist` is maximum distance from node center to points in subtree
//! - lower bound for a node is `d(center, query) - node.max_dist`
//! - if lower bound > current farthest candidate (`tau`), subtree cannot contain a closer point

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::covertree::CoverTree;
use crate::covertree::construct::CoverTreeNode;
use crate::{DistPair, DistanceSearch, Float, KNNHeap};

#[derive(Debug, Clone, Copy)]
struct NodeEntry<'a, F>
where
    F: Float,
{
    lower_bound: F,
    node: &'a CoverTreeNode<F>,
    emit_center: bool,
}

impl<'a, F: Float> PartialEq for NodeEntry<'a, F> {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound
            .partial_cmp(&other.lower_bound)
            .map(|o| o == Ordering::Equal)
            .unwrap_or(false)
    }
}

impl<'a, F: Float> Eq for NodeEntry<'a, F> {}

impl<'a, F: Float> PartialOrd for NodeEntry<'a, F> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<'a, F: Float> Ord for NodeEntry<'a, F> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.lower_bound.partial_cmp(&self.lower_bound).unwrap_or(Ordering::Equal)
    }
}

impl<F: Float> CoverTree<F> {
    /// Find k nearest neighbors using the cover tree pruning logic.
    pub fn search_knn<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, k: usize,
    ) -> Vec<DistPair<F>> {
        if k == 0 {
            return Vec::new();
        }

        let root = match self.root.as_deref() {
            Some(root) => root,
            None => return Vec::new(),
        };

        let mut candidates: KNNHeap<F> = KNNHeap::new(k);
        let mut node_heap: BinaryHeap<NodeEntry<'_, F>> = BinaryHeap::new();

        let root_center_dist = query.query_distance(root.center);
        let root_lower = root_center_dist - root.max_dist;
        node_heap.push(NodeEntry { lower_bound: root_lower, node: root, emit_center: true });

        while let Some(entry) = node_heap.pop() {
            let lower = entry.lower_bound;
            let node = entry.node;
            let emit_center = entry.emit_center;

            let current_tau = candidates.k_distance();

            if lower > current_tau {
                break;
            }

            let d_center = query.query_distance(node.center);
            if emit_center && d_center <= current_tau {
                candidates.insert(DistPair::new(d_center, node.center));
            }

            for &(idx, stored_dist) in node.singletons.iter() {
                if (d_center - stored_dist).abs() <= current_tau {
                    let d = query.query_distance(idx);
                    if d <= current_tau {
                        candidates.insert(DistPair::new(d, idx));
                    }
                }
            }

            let new_tau = candidates.k_distance();

            for child in &node.children {
                let d_child = query.query_distance(child.center);
                let child_lower = d_child - child.max_dist;
                if child_lower <= new_tau {
                    node_heap.push(NodeEntry {
                        lower_bound: child_lower,
                        node: child,
                        emit_center: child.center != node.center,
                    });
                }
            }
        }

        candidates.into_vec()
    }
}
