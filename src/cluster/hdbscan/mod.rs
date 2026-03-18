//! Top-level module for HDBSCAN-specific clustering algorithms.
//!
//! Some utility types such as [`Merge`] are defined in the
//! `hierarchical` module; re-export them here for convenience.

pub use crate::cluster::hierarchical::Merge;

// The various MST and search-based accelerations are now maintained
// in their own submodule so that the `hierarchical` namespace doesn't
// become polluted.  The old hierarchical exports are re-exported from
// `cluster::hdbscan` for compatibility.

pub mod boruvka_searchers_hdbscan;
pub mod buffered_search_hdbscan;
pub mod hdbscan_common;
pub mod hdbscan_prim;
pub mod heap_of_searchers_hdbscan;
pub mod lazy_buffered_search_hdbscan;
pub mod restarting_search_hdbscan;
pub mod slink_hdbscan;

// utility used by multiple algorithms
pub mod extraction;

pub use boruvka_searchers_hdbscan::boruvka_searchers_hdbscan;
pub use buffered_search_hdbscan::buffered_search_hdbscan;
pub use hdbscan_prim::hdbscan_prim;
pub use heap_of_searchers_hdbscan::heap_of_searchers_hdbscan;
pub use lazy_buffered_search_hdbscan::lazy_buffered_search_hdbscan;
pub use restarting_search_hdbscan::restarting_search_hdbscan;
pub use slink_hdbscan::slink_hdbscan;

// re-export the most commonly used items at crate root so callers can
// say `use crate::cluster::hdbscan::heap_of_searchers_hdbscan` etc.
pub use hdbscan_common::HdbscanHierarchy;
