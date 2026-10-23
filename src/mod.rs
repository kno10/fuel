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

#[cfg(feature = "python")]
#[path = "../python/mod.rs"]
mod python;

pub use crate::api::*;

// pull in the crate to link default system BLAS on Linux
// unless an explicit BLAS feature was selected
#[cfg(all(
    target_os = "linux",
    not(any(
        feature = "openblas",
        feature = "openblas-system",
        feature = "openblas-system-static"
    ))
))]
extern crate openblas_src;

// Windows default: static OpenBLAS (slow build)
#[cfg(win_blas_static)]
extern crate openblas_src;
