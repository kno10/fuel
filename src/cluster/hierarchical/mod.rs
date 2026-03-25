//! Hierarchical clustering algorithms and utilities.
//!
//! The `hierarchical` module contains implementations of agglomerative
//! clustering algorithms (currently only AGNES) along with support types
//! such as merge histories and linkage criterions.  By structuring the code
//! in dedicated submodules we avoid polluting the top-level `cluster`
//! namespace and make it easier to extend with additional algorithms in the
//! future.

pub mod agnes;
pub mod anderberg;
pub mod boruvka_searchers_single_link;
pub mod buffered_search_single_link;
pub mod clink;
pub(crate) mod common;
pub mod extraction;
pub mod geometric_nn_chain;
pub mod hacam;
pub mod hausdorff;
pub mod heap_of_searchers_single_link;
pub mod incremental_nn_chain;
pub mod lazy_buffered_search_single_link;
pub mod linkage;
pub mod medoid_linkage;
pub mod muellner;
pub mod set_agnes;
pub mod set_anderberg;
pub mod set_nn_chain;

// algorithm entrypoints
pub use agnes::agnes;
pub use anderberg::anderberg;
pub use boruvka_searchers_single_link::boruvka_searchers_single_link;
pub use buffered_search_single_link::buffered_search_single_link;
pub use clink::{clink, clink_pointer};
pub use geometric_nn_chain::geometric_nn_chain;
pub use hacam::hacam;
pub use hausdorff::hausdorff;
pub use heap_of_searchers_single_link::heap_of_searchers_single_link;
pub use incremental_nn_chain::incremental_nn_chain;
pub use lazy_buffered_search_single_link::lazy_buffered_search_single_link;
pub use muellner::muellner;
pub use nn_chain::nn_chain;
pub use optics_to_hierarchical::optics_to_hierarchical;
pub use pointer::{PointerRepresentation, pointer_to_merge_history};
pub use restarting_search_single_link::restarting_search_single_link;
pub use set_agnes::{set_agnes, set_linkage};
pub use set_anderberg::set_anderberg;
pub use set_nn_chain::set_nn_chain;
pub use slink::{slink, slink_pointer};

pub mod nn_chain;
pub mod optics_to_hierarchical;
pub mod pointer;
pub mod restarting_search_single_link;
pub(crate) mod search_single_link_common;
pub mod slink;

// API level operations
pub use common::{Merge, MergeHistory, PrototypeMergeHistory};
// basic criterion implementations
pub use linkage::average::AverageLinkage;
pub use linkage::centroid::CentroidLinkage;
pub use linkage::complete::CompleteLinkage;
pub use linkage::group_average::GroupAverageLinkage;
pub use linkage::median::MedianLinkage;
pub use linkage::minimum_variance::MinimumVarianceLinkage;
pub use linkage::single::SingleLinkage;
pub use linkage::ward::WardLinkage;
pub use linkage::weighted_average::WeightedAverageLinkage;
// Primary types of linkages
pub use linkage::{GeometricLinkage, Linkage, SetLinkage};

#[cfg(test)]
#[allow(dead_code, unused)]
pub(crate) mod test_utils;

#[cfg(test)]
#[allow(dead_code, unused)]
pub(crate) mod regression_support {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/hierarchical/regression_support.rs"));
}
