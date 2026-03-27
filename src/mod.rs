#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod api;
pub mod cluster;
pub mod data;
pub mod distance;
pub mod distance_matrix;
pub mod evaluation;
pub mod intrinsicdimensionality;
pub mod kernel;
pub mod outlier;
pub mod statistics;

pub mod covertree;
pub mod kd;
pub mod vptree;
pub use crate::api::*;
// convenience data types that are widely used by examples/tests
// `data` module already re-exports the contents of `tabular`, so we can
// refer to `crate::data::TableWithDistance` instead of leaking the private
// `tabular` module.  Consumers still get a short path because of the
// additional root re-export below.
pub use crate::data::TableWithDistance;

// The crate root no longer re‑exports clustering or outlier internals to avoid
// polluting the top-level namespace.  Consumers should go through the
// `cluster` or `outlier` modules directly.

// Note that the following modules are exposed so clients can still reach them:
//   * `crate::cluster`  – all clustering algorithms and types
//   * `crate::outlier`  – outlier detection implementations
//   * `crate::data`, `crate::distance`, etc.  – core building blocks

// (Any additional re-exports should live in the submodules themselves.)
