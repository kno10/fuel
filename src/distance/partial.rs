use num_traits::Float;

/// Distance operations that expose both full-distance and axis-distance helpers.
pub trait PartialDistance<F: Float + Copy> {
    /// Distance between two points represented as slices.
    fn distance(&self, a: &[F], b: &[F]) -> F;

    /// Distance penalty incurred when moving `delta` along one axis.
    ///
    /// The default implementation is appropriate for metrics where the
    /// contribution of an axis can be treated independently (e.g. Manhattan
    /// distance or squared Euclidean distance). For metrics such as Euclidean
    /// distance, this will be used as a safe but potentially loose lower bound.
    fn axis_distance(&self, delta: F) -> F {
        delta.abs()
    }

    /// Combine two per-axis distance contributions into a (lower) bound on the
    /// full distance.
    ///
    /// The default implementation corresponds to an L-infinity bound (max).
    /// Some metrics can produce tighter bounds by summing contributions (e.g.
    /// squared Euclidean distance).
    fn combine_axis_distances(&self, a: F, b: F) -> F {
        a.max(b)
    }
}
