use crate::DistanceData;
use num_traits::Float;

// `rayon` is only required when the `parallel` feature is enabled.  The
// benchmark disables this feature to measure single-threaded performance.
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Helper that returns the starting offset in the flattened triangle for
/// row `i` (i.e. the number of elements in all rows before row `i`).
#[inline]
pub const fn triangle_size(i: usize) -> usize {
    i * (i - 1) / 2
}

/// Compute the *lower triangular* distance matrix for a data set.
///
/// The returned vector contains the distances for pairs `(i,j)` with
/// `0 <= j < i < n` in row-major order.  The length of the resulting
/// vector is `n*(n-1)/2` where `n = data.size()`.
///
/// Parallelisation is performed on the outer index using `rayon`.
pub fn lower_triangular_matrix<D, S>(data: &D) -> Vec<S>
where
    D: DistanceData<S> + Sync,
    S: Float + Send,
{
    let n = data.size();
    if n < 2 {
        return Vec::new();
    }

    #[cfg(feature = "parallel")]
    {
        (1..n)
            .into_par_iter()
            .flat_map_iter(|i| (0..i).map(move |j| data.distance(i, j)))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut out = Vec::with_capacity(n.saturating_sub(1) * n / 2);
        for i in 1..n {
            for j in 0..i {
                out.push(data.distance(i, j));
            }
        }
        out
    }
}
