#![allow(unused)]
//! Metrics built on contingency tables that are intended for external use.
//!
//! This submodule contains the types moved out of `cluster::mod.rs` when the
//! evaluation logic was split into smaller files.  Keeping them in an `external`
//! namespace emphasises that they are high‑level utilities.

pub mod assignment;
pub mod bcubed;
pub mod contingency_table;
pub mod maximum_matching_accuracy;
pub mod mutual_information;
pub mod pair_counting;
pub mod pair_sets_index;
pub mod set_matching;

pub use bcubed::BCubed;
pub use contingency_table::ClusterContingencyTable;
pub use maximum_matching_accuracy::MaximumMatchingAccuracy;
pub use mutual_information::Entropy;
pub use pair_counting::PairCounting;
pub use pair_sets_index::PairSetsIndex;
pub use set_matching::SetMatchingPurity;
