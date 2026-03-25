use crate::Float;

/// Distance bounds per axis, as used for the k-d-tree.
///
/// Partial distances can support different coordinate type and distance type
/// (e.g. `f32` input, `f64` output). Implementers are expected to provide a
/// full distance implementation via `DistanceFunction` in addition to partial
/// distance bounds.
pub trait PartialDistance<N: Float, F: Float> {
    /// Distance penalty incurred when moving `delta` along one axis.
    fn axis_distance(&self, delta: N) -> F;

    /// Combine two per-axis distance contributions into a (lower) bound on the
    /// full distance.
    fn combine_axis_distances(&self, a: F, b: F) -> F;
}
