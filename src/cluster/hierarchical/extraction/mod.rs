//! Cluster extraction utilities for hierarchical merge histories.

mod by_height;
mod by_number_of_clusters;
mod common;
mod hdbscan;

pub use by_height::cut_dendrogram_by_height;
pub use by_number_of_clusters::cut_dendrogram_by_number_of_clusters;
pub use hdbscan::{
    ExtractedHierarchy, HdbscanHierarchyExtractionResult, HierarchyNode,
    extract_clusters_with_noise, extract_hdbscan_hierarchy, extract_hdbscan_hierarchy_hdbscan,
    extract_simplified_hierarchy, extract_simplified_hierarchy_hdbscan,
};
