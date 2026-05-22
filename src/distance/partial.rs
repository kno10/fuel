use crate::Float;

/// Distance bounds per axis, as used for the k-d-tree.
///
/// In addition, this allows operating in two different domains:
/// regular distance, and a bounds domain. For Euclidean distance,
/// using the squared distances as bounds allows saving computations
/// from going back and forth between the two domains.
pub trait PartialDistance<N: Float, F: Float> {
    /// Distance penalty incurred when moving `delta` along one axis.
    fn axis_distance(&self, delta: N) -> F;

    /// Convert a full distance value into this partial bound space.
    ///
    /// For Euclidean raw-bounds, this is squared distance; otherwise identity.
    fn distance_to_range_bound(&self, distance: F) -> F { distance }

    /// Convert a bound value back into regular distance units.
    ///
    /// For Euclidean raw-bounds, this is sqrt; otherwise identity.
    fn range_bound_to_distance(&self, bound: F) -> F { bound }

    /// Update lower bound when one axis contribution is replaced, ideally O(1).
    fn replace_axis_distance(
        &self, current: F, axis: usize, old_axis: F, new_axis: F, axis_bounds: &[F],
    ) -> F;
}

impl<N, F, D> PartialDistance<N, F> for &D
where
    N: Float,
    F: Float,
    D: PartialDistance<N, F> + ?Sized,
{
    fn axis_distance(&self, delta: N) -> F { (**self).axis_distance(delta) }
    fn distance_to_range_bound(&self, distance: F) -> F {
        (**self).distance_to_range_bound(distance)
    }
    fn range_bound_to_distance(&self, bound: F) -> F { (**self).range_bound_to_distance(bound) }
    fn replace_axis_distance(
        &self, current: F, axis: usize, old_axis: F, new_axis: F, axis_bounds: &[F],
    ) -> F {
        (**self).replace_axis_distance(current, axis, old_axis, new_axis, axis_bounds)
    }
}

impl<N, F, D> PartialDistance<N, F> for Box<D>
where
    N: Float,
    F: Float,
    D: PartialDistance<N, F> + ?Sized,
{
    fn axis_distance(&self, delta: N) -> F { (**self).axis_distance(delta) }
    fn distance_to_range_bound(&self, distance: F) -> F {
        (**self).distance_to_range_bound(distance)
    }
    fn range_bound_to_distance(&self, bound: F) -> F { (**self).range_bound_to_distance(bound) }
    fn replace_axis_distance(
        &self, current: F, axis: usize, old_axis: F, new_axis: F, axis_bounds: &[F],
    ) -> F {
        (**self).replace_axis_distance(current, axis, old_axis, new_axis, axis_bounds)
    }
}
