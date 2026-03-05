/// The VP-Tree structure.
#[derive(Debug, PartialEq)]
pub struct VPTree {
    pts: Vec<usize>, // Point ids
    dis: Vec<f64>,   // Distances
}

/// Interface into a data set for distance calculations.
pub trait DataAccess {
    /// Calculate the distance between two points.
    /// The points are identified by their indices in the data set.
    /// The distance function should be metric symmetric.
    fn distance(&self, a: usize, b: usize) -> f64;

    /// Distance from the current query point.
    fn query_distance(&self, b: usize) -> f64;

    /// Get the size of the data set.
    fn size(&self) -> usize;

    /// Allocate a (mutable) vector of indices for the data set.
    fn iter(&self) -> impl Iterator<Item = usize> {
        (0..self.size()).into_iter()
    }
}

pub fn build_vp_tree<T: DataAccess>(data: &T) -> VPTree {
    let n = data.size();
    let mut tree = VPTree {
        pts: vec![0; n],
        dis: vec![0.0; n],
    };

    fn build_recursive<T: DataAccess>(
        data: &T,
        tree: &mut VPTree,
        indices: &mut [usize],
        start: usize,
        end: usize,
        node: usize,
    ) {
        if start >= end {
            return;
        }

        // Select the first point as the vantage point
        let vp_index = indices[start];
        tree.pts[node] = vp_index;

        // Compute distances from the vantage point
        for i in start + 1..end {
            tree.dis[indices[i]] = data.distance(vp_index, indices[i]);
        }

        // Partition points based on their distances to the vantage point
        let mid = (start + end) / 2;
        indices[start + 1..end].sort_by(|&a, &b| {
            tree.dis[a]
                .partial_cmp(&tree.dis[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Recursively build left and right subtrees
        let left_child = (node << 1) + 1;
        let right_child = (node << 1) + 2;

        build_recursive(data, tree, indices, start + 1, mid, left_child);
        build_recursive(data, tree, indices, mid, end, right_child);
    }

    let mut indices: Vec<usize> = data.iter().collect();
    build_recursive(data, &mut tree, &mut indices, 0, n, 0);

    tree
}