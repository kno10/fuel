#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unreadable_literal,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation
)]

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
