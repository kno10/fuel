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
pub mod boruvka_searchers_hdbscan;
pub mod boruvka_searchers_single_link;
pub mod buffered_search_single_link;
pub mod clink;
pub(crate) mod common;
pub mod extraction;
pub mod hacam;
mod hdbscan_common;
pub mod hdbscan_linear_memory;
pub mod heap_of_searchers_hdbscan;
pub mod heap_of_searchers_single_link;
pub mod incremental_nn_chain;
pub mod linear_memory_nn_chain;
pub mod linkage;
pub mod medoid_linkage;
pub mod minimax;
pub mod minimax_anderberg;
pub mod minimax_nn_chain;
pub mod muellner;
pub(crate) mod nn_cache;
pub mod nn_chain;
pub mod optics_to_hierarchical;
pub mod pointer;
pub mod restarting_search_hdbscan;
pub mod restarting_search_single_link;
mod search_single_link_common;
pub mod slink;
pub mod slink_hdbscan_linear_memory;

pub use agnes::agnes;
pub use anderberg::anderberg;
pub use boruvka_searchers_hdbscan::boruvka_searchers_hdbscan;
pub use boruvka_searchers_single_link::boruvka_searchers_single_link;
pub use buffered_search_single_link::buffered_search_single_link;
pub use clink::{clink, clink_pointer};
pub use common::{Merge, MergeHistory, PrototypeMerge, PrototypeMergeHistory};
pub use extraction::{
    ExtractedHierarchy, HdbscanHierarchyExtractionResult, HierarchyNode,
    extract_clusters_with_noise, extract_hdbscan_hierarchy, extract_hdbscan_hierarchy_hdbscan,
    extract_simplified_hierarchy, extract_simplified_hierarchy_hdbscan,
};
pub use extraction::{cut_dendrogram_by_height, cut_dendrogram_by_number_of_clusters};
pub use hacam::{HacamVariant, hacam};
pub use hdbscan_common::HdbscanHierarchy;
pub use hdbscan_linear_memory::hdbscan_linear_memory;
pub use heap_of_searchers_hdbscan::heap_of_searchers_hdbscan;
pub use heap_of_searchers_single_link::heap_of_searchers_single_link;
pub use incremental_nn_chain::incremental_nn_chain;
pub use linear_memory_nn_chain::linear_memory_nn_chain;
pub use linkage::{GeometricLinkage, Linkage};
pub use medoid_linkage::medoid_linkage;
pub use minimax::minimax;
pub use minimax_anderberg::minimax_anderberg;
pub use minimax_nn_chain::minimax_nn_chain;
pub use muellner::muellner;
pub use nn_chain::nn_chain;
pub use optics_to_hierarchical::optics_to_hierarchical;
pub use pointer::{PointerRepresentation, pointer_to_merge_history};
pub use restarting_search_hdbscan::restarting_search_hdbscan;
pub use restarting_search_single_link::restarting_search_single_link;
pub use slink::{slink, slink_pointer};
pub use slink_hdbscan_linear_memory::{
    slink_hdbscan_linear_memory, slink_hdbscan_linear_memory_pointer,
};
// re-export common linkage types for convenience
pub use linkage::{
    AverageLinkage, CentroidLinkage, CompleteLinkage, FlexibleBetaLinkage, GroupAverageLinkage,
    MedianLinkage, MinimumVarianceLinkage, SingleLinkage, WardLinkage, WeightedAverageLinkage,
};

#[cfg(test)]
pub(crate) mod regression_support;

#[cfg(test)]
mod regression;
