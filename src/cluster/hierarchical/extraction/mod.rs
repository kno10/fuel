//! Cluster extraction utilities for hierarchical merge histories.

mod by_height;
mod by_number_of_clusters;
mod common;

pub use by_height::cut_dendrogram_by_height;
pub use by_number_of_clusters::cut_dendrogram_by_number_of_clusters;
