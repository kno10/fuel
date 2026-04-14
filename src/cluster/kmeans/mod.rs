//! k-means clustering algorithms.
//!
//! The port starts with the Lloyd baseline and the corresponding initializers.
//! The more advanced variants remain staged here as commented TODOs so the
//! port can grow in a controlled order.

mod elkan;
mod exponion;
mod fuzzy_cmeans;
mod hamerly;
mod hartigan_wong;
pub mod init;
mod kgeometric;
mod kgeometric_sh;
mod kgmedians;
mod kharmonic;
mod kmedians;
mod lloyd;
mod lloyd_naive;
mod macqueen;
mod selkan;
mod shallot;
mod shamerly;
mod spherical;
mod tkmeans;
pub mod util;

pub use self::elkan::*;
pub use self::exponion::*;
pub use self::fuzzy_cmeans::*;
pub use self::hamerly::*;
pub use self::hartigan_wong::*;
pub use self::init::*;
pub use self::kgeometric::*;
pub use self::kgeometric_sh::*;
pub use self::kgmedians::*;
pub use self::kharmonic::*;
pub use self::kmedians::*;
pub use self::lloyd::*;
pub use self::lloyd_naive::*;
pub use self::macqueen::*;
pub use self::selkan::*;
pub use self::shallot::*;
pub use self::shamerly::*;
pub use self::spherical::*;
pub use self::tkmeans::*;
pub use self::util::{Centers, KMeansResult, compute_fuzzy_loss, compute_loss};
