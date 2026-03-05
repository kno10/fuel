//! Linkage criteria for agglomerative hierarchical clustering.
//!
//! This provides several concrete zero-sized types that implement the various
//! standard linkage methods.  The design is generic so that algorithms such as
//! AGNES can accept *any* `Linkage` implementation, and new methods can be
//! added simply by defining additional types.

pub mod centroid;
pub mod complete;
pub mod flexible_beta;
pub mod group_average;
pub mod median;
pub mod minimum_variance;
pub mod single;
pub mod average;
pub mod ward;
pub mod weighted_average;

pub use centroid::CentroidLinkage;
pub use complete::CompleteLinkage;
pub use flexible_beta::FlexibleBetaLinkage;
pub use group_average::GroupAverageLinkage;
pub use median::MedianLinkage;
pub use minimum_variance::MinimumVarianceLinkage;
pub use single::SingleLinkage;
pub use average::AverageLinkage;
pub use ward::WardLinkage;
pub use weighted_average::WeightedAverageLinkage;

use num_traits::Float;

/// Basic linkage trait corresponds to the Lance–Williams recurrence.
///
/// The previous implementation was hard‑wired to `f64`.  this trait is now
/// parameterised by a floating‑point type `F` so that the same algorithms can
/// be driven by `f32` or any other type implementing `num_traits::Float`.
pub trait Linkage<F: Float>: Copy {
    /// Initialization applied to raw pairwise distances before clustering.
    ///
    /// Some linkage methods (e.g. Ward) operate naturally on squared
    /// distances; in those cases the `initial` step can transform the input
    /// once, avoiding repeated work during the merge updates.  The default
    /// implementation is the identity function.
    fn initial(&self, d: F, _issquare: bool) -> F {
        d
    }

    /// Restore a distance to the conventional scale when recording the result
    /// in the merge history.  This is the inverse of `initial` for methods
    /// that alter the distance scale.
    fn restore(&self, d: F, _issquare: bool) -> F {
        d
    }

    /// Combine two cluster distances according to the chosen method.
    ///
    /// - `sizex`/`sizey`: sizes of the two clusters about to be merged.
    /// - `dx`/`dy`: their respective distances to a third candidate cluster `j`.
    /// - `sizej`: size of that candidate cluster.
    /// - `dxy`: current distance between the two clusters being merged.
    fn combine(&self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F) -> F;
}

/// Extended functionality for geometric linkages used in vector-based
/// algorithms (e.g. Ward) that can aggregate cluster centroids.
pub trait GeometricLinkage<F: Float>: Linkage<F> {
    /// Merge two cluster centres, returning the centre of the combined cluster.
    fn merge(&self, x: &[F], sizex: usize, y: &[F], sizey: usize) -> Vec<F>;

    /// Compute the distance between two aggregated clusters.
    fn linkage(&self, x: &[F], sizex: usize, y: &[F], sizey: usize) -> F;

    /// Restore a linkage value to the original scale.  By default this is the
    /// identity function, but some methods (notably Ward) differ from the
    /// `restore` defined on `Linkage` by a constant factor.
    fn restore_linkage(&self, d: F, _issquare: bool) -> F {
        d
    }
}
