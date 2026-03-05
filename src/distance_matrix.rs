use crate::DataAccess;
use rayon::prelude::*;

/// Compute the *lower triangular* distance matrix for a data set.
///
/// The returned vector contains the distances for pairs `(i,j)` with
/// `0 <= j < i < n` in row-major order.  The length of the resulting
/// vector is `n*(n-1)/2` where `n = data.size()`.
///
/// Parallelisation is performed on the outer index using `rayon`.
pub fn lower_triangular_matrix<D>(data: &D) -> Vec<f64>
where
    D: DataAccess + Sync,
{
    let n = data.size();
    if n < 2 {
        return Vec::new();
    }

    (1..n)
        .into_par_iter()
        .flat_map_iter(|i| (0..i).map(move |j| data.distance(i, j)))
        .collect()
}

/// Helper that returns the starting offset in the flattened triangle for
/// row `i` (i.e. the number of elements in all rows before row `i`).
#[inline]
pub const fn triangle_size(i: usize) -> usize {
    i * (i - 1) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple `DataAccess` implementation for testing.
    struct Dummy {
        pts: Vec<f64>,
    }

    impl DataAccess for Dummy {
        fn distance(&self, a: usize, b: usize) -> f64 {
            (self.pts[a] - self.pts[b]).abs()
        }

        fn query_distance(&self, _b: usize) -> f64 {
            unimplemented!()
        }

        fn size(&self) -> usize {
            self.pts.len()
        }
    }

    #[test]
    fn lower_triangular_simple() {
        let d = Dummy {
            pts: vec![0.0, 1.0, 3.0, 6.0],
        };
        let mat = lower_triangular_matrix(&d);
        // pairs in order: (1,0),(2,0),(2,1),(3,0),(3,1),(3,2)
        assert_eq!(mat, vec![1.0, 3.0, 2.0, 6.0, 5.0, 3.0]);
    }
}
