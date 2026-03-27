use crate::covertree::CoverTree;
use crate::{DistPair, DistanceSearch, Float};

impl<F: Float> CoverTree<F> {
    /// Find all points within radius of the query point (unsorted order).
    pub fn search_range_unsorted<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, radius: F, mut callback: impl FnMut(DistPair<F>),
    ) {
        let root = match self.root.as_deref() {
            Some(root) => root,
            None => return,
        };

        let mut stack = Vec::new();
        stack.push((root, true));

        while let Some((node, emit_center)) = stack.pop() {
            let d_center = query.query_distance(node.center);
            if d_center - node.max_dist > radius {
                continue;
            }

            if emit_center && d_center <= radius {
                callback(DistPair::new(d_center, node.center));
            }

            for child in &node.children {
                let dist = (d_center - child.parent_dist).abs();
                if dist - child.max_dist <= radius {
                    stack.push((child, child.center != node.center));
                }
            }

            for &(idx, s_dist) in node.singletons.iter() {
                if (d_center - s_dist).abs() <= radius {
                    let d = query.query_distance(idx);
                    if d <= radius {
                        callback(DistPair::new(d, idx));
                    }
                }
            }
        }
    }

    /// Range search with result sorting by distance.
    pub fn search_range<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, radius: F, mut callback: impl FnMut(DistPair<F>),
    ) {
        let mut result = Vec::new();
        self.search_range_unsorted(query, radius, |pair| result.push(pair));
        result.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        for pair in result {
            callback(pair);
        }
    }
}
