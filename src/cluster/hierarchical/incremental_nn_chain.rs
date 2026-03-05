use num_traits::Float;

use super::common::MergeHistory;
use super::linear_memory_nn_chain::linear_memory_nn_chain;
use super::linkage::GeometricLinkage;

/// Incremental nearest-neighbor chain clustering for vector data.
///
/// This Rust port currently provides the exact clustering behavior through the
/// same merge process as linear-memory NN-chain, but without index-accelerated
/// incremental priority search.
#[must_use]
pub fn incremental_nn_chain<F: Float, L: GeometricLinkage<F> + Copy>(
    vectors: &[Vec<F>],
    linkage: L,
    is_squared: bool,
) -> MergeHistory<F> {
    linear_memory_nn_chain(vectors, linkage, is_squared)
}

#[cfg(test)]
mod tests {
    use crate::cluster::hierarchical::linkage::WardLinkage;
    use crate::cluster::hierarchical::{incremental_nn_chain, linear_memory_nn_chain};

    #[test]
    fn incremental_nn_chain_matches_linear_memory_variant() {
        let points = vec![
            vec![0.0, 0.0],
            vec![0.2, 0.1],
            vec![3.0, 3.0],
            vec![3.1, 3.2],
            vec![10.0, 10.0],
        ];

        let a = incremental_nn_chain(&points, WardLinkage, false);
        let b = linear_memory_nn_chain(&points, WardLinkage, false);
        assert_eq!(a, b);
    }
}
