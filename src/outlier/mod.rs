pub mod baseline;
pub mod common;
pub mod knn;
pub mod lof;
pub mod r#loop;
pub mod odin;
pub mod weighted_knn;

// convenience exports: users can refer to `crate::outlier::lof_outlier_scores` etc.
// without having to traverse the submodule explicitly.

pub use self::baseline::{
    distance_from_center_outlier_scores, distance_from_origin_outlier_scores,
    random_outlier_scores, zero_outlier_scores,
};

pub use self::baseline::BaselineOutlierScore;
pub use self::knn::{KnnOutlierScore, knn_outlier_scores};
pub use self::lof::{LofOutlierScore, lof_outlier_scores};
pub use self::r#loop::{LoopOutlierScore, loop_outlier_scores};
pub use self::odin::{OdinOutlierScore, odin_outlier_scores};
pub use self::weighted_knn::{WeightedKnnOutlierScore, weighted_knn_outlier_scores};
