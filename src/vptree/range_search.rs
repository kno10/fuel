use std::cmp::Ordering;

use num_traits::Float;

use crate::DataAccess;

use super::{DistPair, VPTree};

impl<F: Float> VPTree<F> {
    /// Find all points within radius r of the query point, without sorting.
    pub fn search_range_unsorted<D: DataAccess>(
        &self,
        data: &D,
        radius: F,
        mut callback: impl FnMut(DistPair<F>),
    ) {
        self.search_range_recursive(data, radius, 0, self.points.len(), &mut callback);
    }

    /// Find all points within radius r of the query point
    pub fn search_range<D: DataAccess>(
        &self,
        data: &D,
        radius: F,
        mut callback: impl FnMut(DistPair<F>),
    ) {
        let mut result = Vec::new();
        self.search_range_unsorted(data, radius, |pair| result.push(pair));

        // Sort results by distance
        result.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        for pair in result {
            callback(pair);
        }
    }

    /// Recursively search for points within radius
    fn search_range_recursive<D: DataAccess, C: FnMut(DistPair<F>)>(
        &self,
        data: &D,
        radius: F,
        left: usize,
        right: usize,
        callback: &mut C,
    ) {
        let node_idx = left;
        let vp = self.points[node_idx];

        // Distance to vantage point
        let d = F::from(data.query_distance(vp as usize))
            .expect("distance cannot be represented by target float type");

        // Add vantage point if within radius
        if d <= radius {
            callback(DistPair::new(d, vp as usize));
        }

        if left + 1 >= right {
            return;
        }

        let mid = usize::midpoint(left, right);

        // Check if we need to search the left subtree
        if left + 1 < mid {
            let left_child = left + 1;
            let child = self.bounds[left_child];
            if child.lower <= d + radius && d - radius <= child.upper {
                self.search_range_recursive(data, radius, left + 1, mid, callback);
            }
        }

        // Check if we need to search the right subtree
        if mid < right {
            let child = self.bounds[mid];
            if child.lower <= d + radius && d - radius <= child.upper {
                self.search_range_recursive(data, radius, mid, right, callback);
            }
        }
    }
}
