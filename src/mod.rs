#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod api;
pub mod cluster;
pub mod distance;
pub mod evaluation;
pub mod intrinsicdimensionality;
pub mod kernel;
pub mod math;
pub mod outlier;
pub mod search;
pub mod statistics;

//#[cfg(feature = "python")]
#[path = "../python/mod.rs"]
mod python;

pub use crate::api::*;
