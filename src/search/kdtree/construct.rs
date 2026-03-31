use std::cmp::Ordering;

use crate::search::kdtree::{KdTree, kdsize};
use crate::{Float, VectorData};

impl<C> KdTree<C>
where
    C: Float,
{
    /// Build a new tree from the given point set using the supplied split heuristic.
    pub fn new<P, S>(data: &P, strategy: S) -> Self
    where
        P: VectorData<C> + ?Sized,
        S: crate::search::kdtree::split::SplitStrategy<C, P>,
    {
        let size = data.len();
        assert!(size <= kdsize::MAX as usize, "data size exceeds kdsize capacity");

        let dims = data.dims();
        assert!(size == 0 || dims > 0, "cannot index zero-dimensional points");

        let mut tree = Self {
            points: vec![0; size],
            split_axes: vec![0; size],
            split_values: vec![C::zero(); size],
        };

        if size > 0 {
            let mut indices: Vec<usize> = (0..size).collect();
            tree.build_recursive(data, &mut indices, 0, size, 0, &strategy);
        }

        tree
    }

    fn build_recursive<P, S>(
        &mut self, data: &P, indices: &mut [usize], left: usize, right: usize, depth: usize,
        strategy: &S,
    ) where
        P: VectorData<C> + ?Sized,
        S: crate::search::kdtree::split::SplitStrategy<C, P>,
    {
        if left >= right {
            return;
        }

        let node_idx = usize::midpoint(left, right);
        let axis = strategy.choose_axis(data, &indices[left..right], depth);
        assert!(axis < data.dims(), "split axis must be in bounds");

        let range = &mut indices[left..right];
        let median = node_idx - left;
        range.select_nth_unstable_by(median, |a, b| {
            data.point(*a)[axis].partial_cmp(&data.point(*b)[axis]).unwrap_or(Ordering::Equal)
        });

        let median_idx = indices[node_idx];
        self.points[node_idx] = median_idx as kdsize;
        self.split_axes[node_idx] = axis as u16;
        self.split_values[node_idx] = data.point(median_idx)[axis];

        self.build_recursive(data, indices, left, node_idx, depth + 1, strategy);
        self.build_recursive(data, indices, node_idx + 1, right, depth + 1, strategy);
    }
}
