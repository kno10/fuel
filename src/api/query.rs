use crate::api::float::Float;

/// A query that can be updated to point at different dataset indices.
pub trait IndexQuery<F: Float>: DistanceSearch<F> {
    /// Update the query to use the given dataset index.
    fn set_index(&mut self, idx: usize);

    /// Update the query and return it for chaining.
    fn with_index(mut self, idx: usize) -> Self
    where
        Self: Sized,
    {
        self.set_index(idx);
        self
    }
}

/// A query that can be updated to use explicit coordinates.
pub trait CoordinateQuery<C: Float, F: Float>: DistanceSearch<F> + CoordinateSearch<C, F> {
    /// Update the query to use the given coordinates.
    fn set_coordinates(&mut self, coords: &[C]);

    /// Update the query and return it for chaining.
    fn with_coordinates(mut self, coords: &[C]) -> Self
    where
        Self: Sized,
    {
        self.set_coordinates(coords);
        self
    }
}

/// Interface for a running search
pub trait DistanceSearch<F: Float> {
    /// Distance from the (fixed) query point.
    fn query_distance(&self, b: usize) -> F;
}

/// Coordinate-base search interface, for k-d-tree etc.
pub trait CoordinateSearch<C: Float, F: Float> {
    /// Number of dimensions in the query embedding space.
    fn dims(&self) -> usize;

    /// Get the query coordinate for a single axis.
    fn query_coordinate(&self, axis: usize) -> C;

    /// Distance bound from a coordinate delta.
    fn delta_to_distance(&self, delta: C) -> F;

    /// Convert a full distance to this partial bound space.
    fn distance_to_range_bound(&self, distance: F) -> F { distance }

    /// Convert a bound value back into regular distance units.
    fn range_bound_to_distance(&self, bound: F) -> F { bound }

    /// Update lower bound when one axis contribution is replaced.
    fn replace_axis_distance(
        &self, current: F, axis: usize, old_axis: F, new_axis: F, axis_bounds: &[F],
    ) -> F;
}
