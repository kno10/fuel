//! Hierarchical clustering algorithms and utilities.
//!
//! The `hierarchical` module contains implementations of agglomerative
//! clustering algorithms (currently only AGNES) along with support types
//! such as merge histories and linkage criterions.  By structuring the code
//! in dedicated submodules we avoid polluting the top-level `cluster`
//! namespace and make it easier to extend with additional algorithms in the
//! future.

pub mod agnes;
pub(crate) mod common;
pub mod linkage;

pub use agnes::agnes;
pub use common::{Merge, MergeHistory};
pub use linkage::{GeometricLinkage, Linkage};
// re-export common linkage types for convenience
pub use linkage::{
    AverageLinkage, CentroidLinkage, CompleteLinkage, FlexibleBetaLinkage, GroupAverageLinkage,
    MedianLinkage, MinimumVarianceLinkage, SingleLinkage, WardLinkage, WeightedAverageLinkage,
};
