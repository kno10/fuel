//! Hierarchical clustering algorithms and utilities.
//!
//! The `hierarchical` module contains implementations of agglomerative
//! clustering algorithms and supporting utilities.  The current code base
//! exposes several algorithmic approaches with differing data structures and
//! performance trade-offs.
//!
//! - `AGNES` is the historic baseline stored-matrix implementation for generic
//!   Lance-Williams linkages.
//! - `Anderberg` adds a nearest-neighbor cache to the same recurrence, but
//!   `Müllner` is the preferred stored-matrix implementation.
//! - `Müllner` extends Anderberg with a heap of candidate nearest-neighbor
//!   pairs for faster merge selection on many linkage types.
//! - `NNChain` uses a nearest-neighbor chain heuristic and can avoid full
//!   pairwise scans, but may yield different results when the chosen linkage
//!   allows inversions.
//! - `HeapOfSearchersSingleLink`, `BoruvkaSearchersSingleLink`,
//!   `BufferedSearchSingleLink`, `LazyBufferedSearchSingleLink`, and
//!   `RestartingSearchSingleLink` are searcher-based single-link methods for
//!   metric data.  `HeapOfSearchersSingleLink` can be very fast, but it uses
//!   more memory than `RestartingSearchSingleLink`.  `slink` remains the best
//!   general-purpose single-link choice when priority search is unavailable.
//! - `GeometricNNChain` and `INNC` belong to the NN-chain family.  They
//!   are specialized for squared Euclidean geometric linkages and avoid the
//!   full condensed matrix by maintaining cluster centres and merge heights.
//!   `INNC` is an incremental nearest-neighbor chain method that is specialized
//!   for geometric linkages, and squared Euclidean `GroupAverageLinkage` can be
//!   a particularly good fit.
//! - `SetAGNES`, `SetAnderberg`, `SetMuellner` and `SetNNChain` are set-based
//!   approaches that compute distances directly from cluster memberships and
//!   summaries.
//!
//! Not every approach can support every linkage.  Stored-matrix algorithms are
//! generally the most generic for Lance-Williams update formulas, but they
//! require the condensed distance matrix and do not exploit vector centroids.
//! Geometric stored-data approaches are limited to vector-valued data and
//! distance functions compatible with centroid-based updates, so they cannot
//! implement prototype-based or Hausdorff linkages.  Set-based approaches
//! support more exotic linkages but pay the price of iterating over cluster
//! members and maintaining explicit membership summaries.
//!
//! The main stored-matrix progression is `AGNES` to `Anderberg` to `Müllner`.
//! `Anderberg` adds a nearest-neighbor cache to the base AGNES recurrence,
//! while `Müllner` adds a heap-backed candidate retrieval layer on top of the
//! same underlying distance updates.
//!
//! Separate from that stored-matrix chain, `NNChain` is a nearest-neighbor
//! chain heuristic that uses a different merge-selection strategy.  It may
//! produce a different merge order for linkages that allow inversions because
//! a chain closure can occur before the global minimum active pair is merged.
//!
//! `GeometricNNChain` and `INNC` are related to the NN-chain family, but they
//! are specialized for geometric stored-data linkages on vector input.  Those
//! methods avoid the full condensed matrix by maintaining cluster centres and
//! merge heights instead of raw pairwise distances.
//!
//! `SetAnderberg` and `SetNNChain` reuse the same acceleration ideas as the
//! stored-matrix variants, but in a set-based representation that supports
//! linkages like `MedoidLinkage`, `MinimaxLinkage`, `MinimumSumLinkage`, and
//! `HausdorffLinkage`.
//!
//! `INNC` may diverge from strict agglomerative clustering for
//! `CentroidLinkage` and `MedianLinkage` because the spatial bounds used by the
//! searcher are not strict and can accept a suboptimal merge.
//!
//! Overview of linkage support and recommended approaches:
//!
//! - `SingleLinkage`
//!   - for metric data: searcher-based single-link methods
//!     (`HeapOfSearchersSingleLink`, `BufferedSearchSingleLink`,
//!     `LazyBufferedSearchSingleLink`, `RestartingSearchSingleLink`,
//!     `BoruvkaSearchersSingleLink`).
//!   - general purpose: `slink`
//!   - alternative: stored-matrix `Müllner` or set-based.
//! - `CompleteLinkage`
//!   - recommended: stored-matrix `Müllner`.
//!   - alternative: set-based; `clink` is available but not recommended.
//! - `GroupAverageLinkage`, `WeightedAverageLinkage`
//!   - recommended: stored-matrix approaches.
//!   - alternative: set-based.
//! - `CentroidLinkage`, `MedianLinkage`
//!   - recommended: geometric stored-data.
//!   - alternative: stored-matrix.
//! - `WardLinkage`, `MinimumVarianceLinkage`, `MinimumVarianceIncreaseLinkage`,
//!   `MinimumSumSquaresLinkage`
//!   - recommended: geometric stored-data.
//!   - alternative: stored-matrix or set-based.
//! - `FlexibleBetaLinkage`
//!   - supported only by stored-matrix approaches.
//! - `MedoidLinkage`, `MinimaxLinkage`, `MinimumSumLinkage`,
//!   `MinimumSumIncreaseLinkage`, `HausdorffLinkage`
//!   - supported only by set-based approaches; `SetNNChain` is probably the best
//!     choice for these set-based-only linkages.
//!
//! Geometric linkages such as `CentroidLinkage` and `MedianLinkage` require
//! vector data and squared Euclidean distance to preserve centroid updates.
//! Prototype-based and Hausdorff linkages are only viable in set-based
//! implementations.

/// `idsize`` is the type used for instance and cluster ids in several algorithms;
/// it controls the trade-off between memory use and maximum data set size.
/// With u32, we can support up to ~2^31 instances, we will run out of memory long before this.
#[allow(non_camel_case_types)]
pub type idsize = u32;

pub mod agnes;
pub mod anderberg;
pub mod boruvka_searchers_single_link;
pub mod buffered_search_single_link;
pub mod clink;
pub(crate) mod common;
pub mod extraction;
pub mod geometric_nn_chain;
pub mod hausdorff;
pub mod heap_of_searchers_single_link;
pub mod incremental_nn_chain;
pub mod lazy_buffered_search_single_link;
pub mod linkage;
pub mod medoid_linkage;
pub(crate) mod merge_history;
pub mod muellner;
pub mod set_agnes;
pub mod set_anderberg;
pub mod set_muellner;
pub mod set_nn_chain;

// algorithm entrypoints
pub use agnes::agnes;
pub use anderberg::anderberg;
pub use boruvka_searchers_single_link::boruvka_searchers_single_link;
pub use buffered_search_single_link::buffered_search_single_link;
pub use clink::{clink, clink_pointer};
pub use geometric_nn_chain::geometric_nn_chain;
pub use hausdorff::hausdorff;
pub use heap_of_searchers_single_link::heap_of_searchers_single_link;
pub use incremental_nn_chain::incremental_nn_chain;
pub use lazy_buffered_search_single_link::lazy_buffered_search_single_link;
pub use muellner::muellner;
pub use nn_chain::nn_chain;
pub use optics_to_hierarchical::optics_to_hierarchical;
pub use pointer::{PointerRepresentation, pointer_to_merge_history};
pub use restarting_search_single_link::restarting_search_single_link;
pub use set_agnes::set_agnes;
pub use set_anderberg::{set_anderberg, set_anderberg as hacam};
pub use set_muellner::set_muellner;
pub use set_nn_chain::set_nn_chain;
pub use slink::{slink, slink_pointer};

pub mod nn_chain;
pub mod optics_to_hierarchical;
pub mod pointer;
pub mod restarting_search_single_link;
pub(crate) mod search_single_link_common;
pub mod slink;

// API level operations
// basic criterion implementations
pub use linkage::centroid::CentroidLinkage;
pub use linkage::complete::CompleteLinkage;
pub use linkage::flexible_beta::FlexibleBetaLinkage;
pub use linkage::group_average::GroupAverageLinkage;
pub use linkage::hausdorff::HausdorffLinkage;
pub use linkage::median::MedianLinkage;
pub use linkage::medoid::MedoidLinkage;
pub use linkage::minimax::MinimaxLinkage;
pub use linkage::minimum_sum::MinimumSumLinkage;
pub use linkage::minimum_sum_increase::MinimumSumIncreaseLinkage;
pub use linkage::minimum_sum_squares::MinimumSumSquaresLinkage;
pub use linkage::minimum_variance::MinimumVarianceLinkage;
pub use linkage::minimum_variance_increase::MinimumVarianceIncreaseLinkage;
pub use linkage::single::SingleLinkage;
pub use linkage::ward::WardLinkage;
pub use linkage::weighted_average::WeightedAverageLinkage;
// Primary types of linkages
pub use linkage::{GeometricLinkage, Linkage, SetLinkage};
pub(crate) use merge_history::Builder;
pub use merge_history::{Merge, MergeHistory};
#[cfg(test)]
pub(crate) mod test;
