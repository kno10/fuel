use crate::search::covertree::CoverTree;
use crate::{DistPair, DistanceSearch, Float};

impl<F: Float> CoverTree<F> {
    pub fn search_range_unsorted<Q: DistanceSearch<F> + ?Sized>(
        &self, query: &Q, radius: F, mut callback: impl FnMut(DistPair<F>),
    ) {
        let mut stack = Vec::new();
        stack.push((0, true));

        while let Some((node_idx, emit_center)) = stack.pop() {
            let node = &self.nodes[node_idx as usize];
            let d_center = query.query_distance(node.center);
            if d_center - node.max_dist > radius {
                continue;
            }

            if emit_center && d_center <= radius {
                callback(DistPair::new(d_center, node.center));
            }

            for &child_idx in &node.children {
                let child = &self.nodes[child_idx as usize];
                let dist = (d_center - child.parent_dist).abs();
                if dist - child.max_dist <= radius {
                    stack.push((child_idx, child.center != node.center));
                }
            }

            for singleton in &node.singletons {
                let idx = singleton.index;
                let stored_dist = singleton.distance;
                if (d_center - stored_dist).abs() <= radius {
                    let d = query.query_distance(idx);
                    if d <= radius {
                        callback(DistPair::new(d, idx));
                    }
                }
            }
        }
    }

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

#[cfg(test)]
mod tests {
    use crate::distance::SquaredEuclidean;
    use crate::search::covertree::CoverTree;
    use crate::{CoordinateQuery, DistPair, DistanceData, TableWithDistance};

    fn sample_points() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, 2.0]]
    }

    #[test]
    fn cover_tree_range_finds_close_points() {
        let points = sample_points();
        let data = TableWithDistance::with_distance(&points, SquaredEuclidean);
        let tree = CoverTree::new(&data, 1.3, 0);
        let query = data.query().with_coordinates(&points[0]);

        let result: Vec<DistPair<f64>> = crate::api::RangeSearch::search_range(&tree, &query, 1.01);

        assert!(result.iter().any(|n| n.index == 0));
        assert!(result.iter().any(|n| n.index == 1));
        assert!(result.iter().any(|n| n.index == 2));
        assert!(result.iter().all(|n| n.distance <= 1.01));
    }
}
