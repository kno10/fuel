pub mod first;
pub mod kgeometricpp;
pub mod kmeanspp;
/// Initialization API for k-means
pub mod random;

use std::ops::{AddAssign, MulAssign, SubAssign};

use crate::cluster::kmeans::util::*;
use crate::{Float, VectorData as Dataset};

/// Trait to choose initial means
pub trait Initialization<N>
where
    N: Float + Copy + AddAssign + SubAssign + MulAssign,
{
    /// Computes distances
    fn uses_distances(&self) -> bool;

    /// Choose initial means
    fn init<A>(&mut self, data: &A, cent: &mut Centers<N>, k: usize)
    where
        A: Dataset<N>;

    /// Choose initial means
    // Callback: initialization that compute all(!) distances may call this callback
    fn init_with_distances<A, F>(
        &mut self, data: &A, cent: &mut Centers<N>, k: usize, callback: Option<F>,
    ) where
        A: Dataset<N>,
        F: FnMut(usize, usize, N);
}

// re-export convenient types
pub use first::FirstK;
pub use kgeometricpp::KGeometricPP;
pub use kmeanspp::KMeansPP;
pub use random::RandomSample;

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::TableWithDistance;
    use crate::distance::Euclidean;

    #[test]
    fn firstk_initialization() {
        let points = vec![vec![1., 2.], vec![3., 4.], vec![5., 6.]];
        let ds = TableWithDistance::<f64, _, _, f64>::with_distance(&points, Euclidean);
        let mut init = FirstK::new();
        assert!(!init.uses_distances());
        let mut cent = Centers::new(2, 2);
        init.init(&ds, &mut cent, 2);
        // first two rows should be copied
        assert_eq!(cent.center(0), &[1., 2.]);
        assert_eq!(cent.center(1), &[3., 4.]);
    }

    #[test]
    fn random_sample_bounds() {
        let points: Vec<Vec<f64>> =
            (0..3).map(|i| (0..3).map(|j| (i * 3 + j) as f64).collect()).collect();
        let ds = TableWithDistance::<f64, _, _, f64>::with_distance(&points, Euclidean);
        let rng = StdRng::seed_from_u64(42);
        let mut init = RandomSample::new(rng);
        assert!(!init.uses_distances());
        let mut cent = Centers::new(2, 3);
        init.init(&ds, &mut cent, 2);
        // centroids must come from dataset (values between 0 and 8)
        for i in 0..2 {
            for &v in cent.center(i) {
                assert!((0.0..9.0).contains(&v));
            }
        }
    }

    #[test]
    fn kmeanspp_uses_distances() {
        let points = vec![vec![0., 1.], vec![2., 3.]];
        let ds = TableWithDistance::<f64, _, _, f64>::with_distance(&points, Euclidean);
        let rng = StdRng::seed_from_u64(123);
        let mut init = KMeansPP::new(rng);
        assert!(init.uses_distances());
        let mut cent = Centers::new(2, 2);
        // callback increments a counter
        let mut count = 0;
        init.init_with_distances::<_, _>(
            &ds,
            &mut cent,
            2,
            Some(|_, _, _| {
                count += 1;
            }),
        );
        assert!(count > 0);
    }

    #[test]
    fn kgeometricpp_uses_distances_and_in_dataset() {
        let points: Vec<Vec<f64>> =
            (0..3).map(|i| (0..3).map(|j| (i * 3 + j) as f64).collect()).collect();
        let ds = TableWithDistance::<f64, _, _, f64>::with_distance(&points, Euclidean);
        let rng = StdRng::seed_from_u64(42);
        let mut init = KGeometricPP::new(rng);
        assert!(init.uses_distances());
        let mut cent = Centers::new(2, 3);
        init.init(&ds, &mut cent, 2);
        for i in 0..2 {
            for &v in cent.center(i) {
                assert!((0.0..9.0).contains(&v));
            }
        }
    }
}
