use crate::{Float, VectorData};

/// A strategy for choosing the splitting axis when building a KD-tree.
pub trait SplitStrategy<F, P>
where
    F: Float,
    P: VectorData<F> + ?Sized,
{
    /// Decide which axis to split on for the current subset.
    fn choose_axis(&self, data: &P, candidates: &[usize], depth: usize) -> usize;
}

/// Cycle through axes in round-robin order.
#[derive(Clone, Copy, Default, Debug)]
pub struct AxisCycleSplit;

impl<F, P> SplitStrategy<F, P> for AxisCycleSplit
where
    F: Float,
    P: VectorData<F> + ?Sized,
{
    fn choose_axis(&self, data: &P, _candidates: &[usize], depth: usize) -> usize {
        let dims = data.dims();
        assert!(dims > 0, "cannot split zero-dimensional data");
        depth % dims
    }
}

/// Always pick the axis with the largest span between min and max.
#[derive(Clone, Copy, Default, Debug)]
pub struct LargestSpreadSplit;

impl<F, P> SplitStrategy<F, P> for LargestSpreadSplit
where
    F: Float,
    P: VectorData<F> + ?Sized,
{
    fn choose_axis(&self, data: &P, candidates: &[usize], _depth: usize) -> usize {
        let dims = data.dims();
        assert!(dims > 0, "cannot split zero-dimensional data");
        if candidates.is_empty() {
            return 0;
        }

        let mut best_axis = 0;
        let mut best_span = 0.0;
        for axis in 0..dims {
            let base = data.point(candidates[0])[axis].to_f64().unwrap_or(0.0);
            let mut minv = base;
            let mut maxv = base;
            for &idx in candidates.iter().skip(1) {
                let value = data.point(idx)[axis].to_f64().unwrap_or(0.0);
                if value < minv {
                    minv = value;
                }
                if value > maxv {
                    maxv = value;
                }
            }
            let span = maxv - minv;
            if axis == 0 || span > best_span {
                best_span = span;
                best_axis = axis;
            }
        }

        best_axis
    }
}

/// Pick the axis with the largest observed variance.
#[derive(Clone, Copy, Default, Debug)]
pub struct MaxVarianceSplit;

impl<F, P> SplitStrategy<F, P> for MaxVarianceSplit
where
    F: Float,
    P: VectorData<F> + ?Sized,
{
    fn choose_axis(&self, data: &P, candidates: &[usize], _depth: usize) -> usize {
        let dims = data.dims();
        assert!(dims > 0, "cannot split zero-dimensional data");
        if candidates.is_empty() {
            return 0;
        }

        let n = candidates.len() as f64;
        let mut sums = vec![0.0; dims];
        let mut sums_sq = vec![0.0; dims];

        for &idx in candidates {
            let point = data.point(idx);
            for axis in 0..dims {
                let value = point[axis].to_f64().unwrap_or(0.0);
                sums[axis] += value;
                sums_sq[axis] += value * value;
            }
        }

        let mut best_axis = 0;
        let mut best_var = f64::NEG_INFINITY;
        for axis in 0..dims {
            let mean = sums[axis] / n;
            let var = sums_sq[axis] / n - mean * mean;
            if axis == 0 || var > best_var {
                best_var = var;
                best_axis = axis;
            }
        }

        best_axis
    }
}

#[cfg(test)]
mod tests {
    use super::{LargestSpreadSplit, MaxVarianceSplit, SplitStrategy};
    use crate::TableWithDistance;
    use crate::distance::Euclidean;

    fn sample_points() -> Vec<Vec<f64>> { vec![vec![0.0, 0.0], vec![1.0, 10.0], vec![1.5, -20.0]] }

    fn sample_points_variance() -> Vec<Vec<f64>> {
        vec![vec![0.0, 0.0], vec![5.0, 0.0], vec![10.0, 0.0], vec![5.0, 0.1]]
    }

    #[test]
    fn largest_spread_prefers_widest_axis() {
        let points = sample_points();
        let split = LargestSpreadSplit;
        let data: TableWithDistance<'_, f64, Vec<f64>, Euclidean, f64> =
            TableWithDistance::with_distance(&points, Euclidean);
        let axis = split.choose_axis(&data, &[0, 1, 2], 0);
        assert_eq!(axis, 1);
    }

    #[test]
    fn max_variance_prefers_noisy_axis() {
        let points = sample_points_variance();
        let split = MaxVarianceSplit;
        let data: TableWithDistance<'_, f64, Vec<f64>, Euclidean, f64> =
            TableWithDistance::with_distance(&points, Euclidean);
        let axis = split.choose_axis(&data, &[0, 1, 2, 3], 0);
        assert_eq!(axis, 0);
    }
}
