//! Linkage criteria for agglomerative hierarchical clustering.
//!
//! This provides several concrete zero-sized types that implement the various
//! standard linkage methods.  The design is generic so that algorithms such as
//! AGNES can accept *any* `Linkage` implementation, and new methods can be
//! added simply by defining additional types.
//!
//! The table below summarises the implemented linkages, their recurrence,
//! whether they can produce inversions, and the approaches they support.
//!
//! | Linkage | Recurrence | Inversions | Supported approaches |
//! |---|---|---|---|
//! | `SingleLinkage` | $\min(dx, dy)$ | no | stored-matrix, set-based |
//! | `CompleteLinkage` | $\max(dx, dy)$ | no | stored-matrix, set-based |
//! | `GroupAverageLinkage` | $\frac{n_x dx + n_y dy}{n_x + n_y}$ | no | stored-matrix, geometric, set-based |
//! | `WeightedAverageLinkage` | $\tfrac{1}{2}(dx + dy)$ | no | stored-matrix |
//! | `CentroidLinkage` | $\frac{n_x dx + n_y dy - \frac{n_x n_y}{n_x + n_y} d_{xy}}{n_x + n_y}$ | yes | stored-matrix, geometric |
//! | `MedianLinkage` | $\tfrac{1}{2}(dx + dy) - \tfrac{1}{4} d_{xy}$ | yes | stored-matrix, geometric |
//! | `WardLinkage` | $\frac{(n_x+n_k)dx + (n_y+n_k)dy - n_k d_{xy}}{n_x+n_y+n_k}$ | no | stored-matrix, geometric, set-based |
//! | `MinimumSumSquaresLinkage` | within-cluster sum of squared deviations | no | stored-matrix, geometric, set-based |
//! | `MinimumVarianceLinkage` | average cluster variance objective | no | stored-matrix, geometric, set-based |
//! | `MinimumVarianceIncreaseLinkage` | variance increase variant | yes | stored-matrix, geometric, set-based |
//! | `FlexibleBetaLinkage` | $\alpha dx + \alpha dy + \beta d_{xy}$ | yes if $\beta < 0$ | stored-matrix |
//! | `MinimaxLinkage` | $\min_{z\in X\cup Y} \max(d(z,X), d(z,Y))$ | no | set-based only |
//! | `HausdorffLinkage` | directed Hausdorff maximum min-distance | yes | set-based only |
//! | `MedoidLinkage` | prototype medoid of union | no | set-based only |
//! | `MinimumSumLinkage` | minimum total distance medoid | no | set-based only |
//! | `MinimumSumIncreaseLinkage` | minimum-sum with correction | no | set-based only |

pub mod centroid;
pub mod complete;
pub mod flexible_beta;
pub mod group_average;
pub mod median;
pub mod minimum_sum_squares;
pub mod minimum_variance;
pub mod minimum_variance_increase;
pub mod single;
pub mod ward;
pub mod weighted_average;

pub mod hausdorff;
pub mod medoid;
pub mod minimax;

// HACAM-specific set-linkage variants
pub mod minimum_sum;
pub mod minimum_sum_increase;

use crate::cluster::hierarchical::idsize;
use crate::{DistanceData, Float, math};

/// Basic linkage trait corresponds to the Lance-Williams recurrence.
///
/// The previous implementation was hard-wired to `f64`.  this trait is now
/// parameterised by a floating-point type `F` so that the same algorithms can
/// be driven by `f32` or any other type implementing `num_traits::Float`.
pub trait Linkage<F: Float>: Copy {
    /// Whether this linkage can produce inversions, i.e. a later merge with a
    /// smaller distance than an earlier merge.
    #[allow(unused)]
    fn can_produce_inversions(&self) -> bool { false }

    /// Initialization applied to raw pairwise distances before clustering.
    ///
    /// Some linkage methods (e.g. Ward) operate naturally on squared
    /// distances; in those cases the `initial` step can transform the input
    /// once, avoiding repeated work during the merge updates.  The default
    /// implementation is the identity function.
    #[allow(unused)]
    fn initial(&self, d: F, issquare: bool) -> F { d }

    /// Restore a distance to the conventional scale when recording the result
    /// in the merge history.  This is the inverse of `initial` for methods
    /// that alter the distance scale.
    #[allow(unused)]
    fn restore(&self, d: F, issquare: bool) -> F { d }

    /// Combine two cluster distances according to the chosen method.
    ///
    /// - `sizex`/`sizey`: sizes of the two clusters about to be merged.
    /// - `dx`/`dy`: their respective distances to a third candidate cluster `j`.
    /// - `sizej`: size of that candidate cluster.
    /// - `dxy`: current distance between the two clusters being merged.
    /// - `heightx`/`heighty`/`heightj`: last merge heights of the three clusters
    ///   in the same distance scale used by the linkage recurrence.
    fn combine(
        &self, sizex: usize, dx: F, sizey: usize, dy: F, sizej: usize, dxy: F, heightx: F,
        heighty: F, heightj: F,
    ) -> F;
}

/// Extended functionality for geometric linkages used in vector-based
/// algorithms (e.g. Ward) that can aggregate cluster centroids.
/// These approaches are sometimes called "stored data" approaches in literature,
/// but mostly limited to (squared) Euclidean distance because they rely on the König-Huygens identity.
pub trait GeometricLinkage<F: Float>: Linkage<F> {
    /// Merge two cluster centres, returning the centre of the combined cluster.
    ///
    /// The default implementation is the weighted mean of cluster vectors,
    /// which is valid for centroid-based geometric linkages such as Ward,
    /// centroid, minimum sum-of-squares and group-average.
    #[allow(unused)]
    fn merge(
        &self, x: &[F], sizex: usize, y: &[F], sizey: usize, heightx: F, heighty: F,
    ) -> Vec<F> {
        let sx = F::from(sizex).unwrap();
        let sy = F::from(sizey).unwrap();
        let mut out = x.to_vec();
        let dim = out.len();
        math::axpby(&mut out, sx / (sx + sy), y, sy / (sx + sy), dim);
        out
    }

    /// Compute the distance between two aggregated clusters.
    fn linkage(&self, x: &[F], sizex: usize, y: &[F], sizey: usize, heightx: F, heighty: F) -> F;

    /// Compute the internal cluster summary height for a newly merged cluster.
    ///
    /// Most geometric linkages can use the same value as the pairwise linkage
    /// between the merged clusters.  Some methods, such as group-average,
    /// require a distinct cluster summary to preserve future distance updates.
    #[allow(unused)]
    fn merge_height(
        &self, x: &[F], sizex: usize, y: &[F], sizey: usize, heightx: F, heighty: F,
    ) -> F {
        self.linkage(x, sizex, y, sizey, heightx, heighty)
    }

    /// Bound factor used when resetting the searcher cutoff for a cluster.
    ///
    /// The default is no additional widening beyond the current best link.
    #[allow(unused)]
    fn cutoff_factor(&self, _size_a: usize) -> F { F::one() }

    /// Threshold for candidate point distances used to prune clusters.
    ///
    /// The default implementation is no looser than the current best link.
    #[allow(unused)]
    fn candidate_threshold(
        &self, min_link: F, _size_a: usize, _size_i: usize, _height_a: F, _height_i: F,
    ) -> F {
        min_link
    }
}

/// Linkage criterion expressed as a function of the *sets* underlying two
/// clusters.
///
/// The implementation is intentionally simple: given two slices of point
/// indices representing cluster membership, return a tuple of the distance
/// between the clusters, an optional prototype index for the merged cluster,
/// and enough per-cluster summary data to avoid re-computing prototypes or
/// accumulated distances during the merge process.
pub trait SetLinkage<D: DistanceData<F>, F: Float, Summary = ()> {
    /// Whether this linkage can produce inversions, i.e. a later merge with a
    /// smaller distance than an earlier merge.
    #[allow(unused)]
    fn can_produce_inversions(&self) -> bool { false }

    /// Summary information that is maintained for each cluster throughout the
    /// clustering process.  The summary is expected to encode protocol-specific
    /// prototypes or statistics that are reused when merging or comparing
    /// clusters.
    fn summarize(data: &D, members: &[idsize]) -> Summary;

    /// Distance between cluster `a` and cluster `b`, plus the summary of the
    /// merged cluster.
    fn cluster_distance(
        data: &D, summary_a: &Summary, summary_b: &Summary, a: &[idsize], b: &[idsize],
    ) -> (F, Summary);

    /// Extract the prototype index for the merged cluster summary.
    ///
    /// If a linkage implementation does not track a prototype, it should
    /// return `usize::MAX`.
    #[allow(unused)]
    fn merged_prototype(summary: &Summary) -> usize { usize::MAX }

    /// Restore a cluster distance to the original data scale.
    ///
    /// Some set-based linkages compute distances on an internal transformed
    /// scale (for example, minimum variance and Ward).  The merge history
    /// should record distances in the same units as the corresponding
    /// `Linkage` implementations.
    #[allow(unused)]
    fn restore(d: F, issquare: bool) -> F { d }
}
