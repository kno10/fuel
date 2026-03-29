use std::cmp::Ordering;

use rand::Rng;
use rand::seq::index;

use super::{Bounds, PrioritySearcher, VPTree, vpsize};
use crate::{DistanceData, Float};

impl<T: Float> VPTree<T> {
    /// Create a new VP-Tree from the given data with improved vantage point selection
    ///
    /// # Panics
    ///
    /// Panics if the input data set is empty.
    pub fn new<D: DistanceData<T>, R: Rng>(data: &D, sample_size: usize, rng: &mut R) -> Self {
        let size = data.size();
        assert!(size > 0, "Data set must contain at least one point.");
        assert!(size <= vpsize::MAX as usize, "Data set size exceeds vpsize capacity");

        let mut tree = Self {
            points: vec![0; size],
            bounds: vec![Bounds::<T>::new(T::nan(), T::nan()); size],
        };
        let mut indices: Vec<usize> = (0..size).collect();

        tree.build_tree(
            data,
            &mut indices,
            0,
            size,
            Bounds::<T>::new(T::nan(), T::nan()),
            sample_size,
            rng,
        );
        tree
    }

    /// Create an incremental priority searcher for a query.
    ///
    /// The returned searcher can be reused across queries with `reset` to avoid
    /// repeated internal reallocations.
    pub fn priority_searcher(&self) -> PrioritySearcher<'_, T> { PrioritySearcher::new(self) }

    /// Recursively build the VP-Tree with sampling for vantage point selection
    #[allow(clippy::too_many_arguments)]
    fn build_tree<D: DistanceData<T>, R: Rng>(
        &mut self, data: &D, indices: &mut [usize], left: usize, right: usize, bounds: Bounds<T>,
        sample_size: usize, rng: &mut R,
    ) {
        assert!(left < right);
        let node_idx = left;
        self.bounds[node_idx] = bounds;

        // When we have only one point, just return a leaf node
        if left + 1 >= right {
            self.points[node_idx] = indices[left] as vpsize;
            return;
        }

        // Select vantage point with sampling if we have enough points
        let vp_idx = if right - left > sample_size && sample_size > 1 {
            Self::choose_vantage_point(data, indices, left, right, sample_size, rng)
        } else {
            indices[left] // Default to first point if sampling not possible
        };

        // Swap the vantage point to the first position
        if indices[left] != vp_idx {
            let vp_pos = indices[left..right].iter().position(|&i| i == vp_idx).unwrap() + left;
            indices.swap(left, vp_pos);
        }

        self.points[node_idx] = vp_idx as vpsize;

        // Partition remaining points based on distance to vantage point
        let mut dists: Vec<(T, usize)> = Vec::with_capacity(right - left - 1);
        for &idx in indices.iter().take(right).skip(left + 1) {
            dists.push((data.distance(vp_idx, idx), idx));
        }

        // Sort by distance
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Update indices array with sorted order
        for (i, &(_, idx)) in dists.iter().enumerate() {
            indices[left + 1 + i] = idx;
        }

        // Find median position
        let mid = usize::midpoint(left, right);

        // Calculate bounds for left and right subtrees
        let mut left_low = T::infinity();
        let mut left_high = T::neg_infinity();
        let mut right_low = T::infinity();
        let mut right_high = T::neg_infinity();

        let left_count = mid.saturating_sub(left + 1);
        for &(d, _) in dists.iter().take(left_count) {
            left_low = left_low.min(d);
            left_high = left_high.max(d);
        }

        for &(d, _) in dists.iter().skip(left_count) {
            right_low = right_low.min(d);
            right_high = right_high.max(d);
        }

        // Recursively build subtrees
        if left + 1 < mid {
            self.build_tree(
                data,
                indices,
                left + 1,
                mid,
                Bounds::<T>::new(left_low, left_high),
                sample_size,
                rng,
            );
        }

        if mid < right {
            self.build_tree(
                data,
                indices,
                mid,
                right,
                Bounds::<T>::new(right_low, right_high),
                sample_size,
                rng,
            );
        }
    }

    /// Choose a vantage point from a sample that maximizes the spread of distances
    fn choose_vantage_point<D: DistanceData<T>, R: Rng>(
        data: &D, indices: &[usize], left: usize, right: usize, sample_size: usize, rng: &mut R,
    ) -> usize {
        // Draw distinct sample positions from the active partition.
        let partition_len = right - left;
        let sample_len = sample_size.min(partition_len);
        let sampled_positions = index::sample(rng, partition_len, sample_len);
        let sample: Vec<usize> =
            sampled_positions.iter().map(|offset| indices[left + offset]).collect();

        // For each candidate, calculate the spread (variance) of distances
        let mut best_spread = T::neg_infinity();
        let mut best_vp = sample[0]; // Default to first point

        for &vp in &sample {
            // Compute distances to other points in the sample
            let mut distances = Vec::with_capacity(sample_size);
            for &p in &sample {
                if p != vp {
                    distances.push(data.distance(vp, p));
                }
            }

            if distances.is_empty() {
                continue;
            }

            // Calculate median without fully sorting.
            let mid = distances.len() / 2;
            let (_, median, _) = distances
                .select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let median = *median;

            // Calculate spread (second moment about the median)
            let spread = distances
                .iter()
                .map(|&d| {
                    let delta = d - median;
                    delta * delta
                })
                .fold(T::zero(), |acc, x| acc + x)
                / T::from(distances.len())
                    .expect("distance sample length cannot be represented by target float type");

            if spread > best_spread {
                best_spread = spread;
                best_vp = vp;
            }
        }

        best_vp
    }
}
