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

    /// simple in-memory dataset: n rows, d cols stored row-major
    struct SimpleDataset<N> {
        data: Vec<N>,
        n: usize,
        d: usize,
    }
    impl<N: Copy> SimpleDataset<N> {
        fn new(data: Vec<N>, n: usize, d: usize) -> Self {
            assert_eq!(n * d, data.len());
            SimpleDataset { data, n, d }
        }
    }
    impl<N: Copy> crate::Data for SimpleDataset<N> {
        fn len(&self) -> usize { self.n }
    }

    impl<N: Copy> Dataset<N> for SimpleDataset<N> {
        fn nrows(&self) -> usize { self.n }
        fn ncols(&self) -> usize { self.d }
        fn dims(&self) -> usize { self.d }
        fn point(&self, idx: usize) -> &[N] {
            let start = idx * self.d;
            &self.data[start..start + self.d]
        }
        fn load_into(&self, i: usize, vec: &mut [N], d: usize) {
            let start = i * self.d;
            vec[..d].copy_from_slice(&self.data[start..start + d]);
        }
    }

    fn make_centroids<N: Float>(k: usize, d: usize) -> Centers<N> { Centers::new(k, d) }

    #[test]
    fn firstk_initialization() {
        let ds = SimpleDataset::new(vec![1., 2., 3., 4., 5., 6.], 3, 2);
        let mut init = FirstK::new();
        assert!(!init.uses_distances());
        let mut cent = make_centroids::<f64>(2, 2);
        init.init(&ds, &mut cent, 2);
        // first two rows should be copied
        assert_eq!(cent.center(0), &[1., 2.]);
        assert_eq!(cent.center(1), &[3., 4.]);
    }

    #[test]
    fn random_sample_bounds() {
        let ds = SimpleDataset::new((0..9).map(|x| x as f64).collect(), 3, 3);
        let rng = StdRng::seed_from_u64(42);
        let mut init = RandomSample::new(rng);
        assert!(!init.uses_distances());
        let mut cent = make_centroids::<f64>(2, 3);
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
        let ds = SimpleDataset::new(vec![0., 1., 2., 3.], 2, 2);
        let rng = StdRng::seed_from_u64(123);
        let mut init = KMeansPP::new(rng);
        assert!(init.uses_distances());
        let mut cent = make_centroids::<f64>(2, 2);
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
        let ds = SimpleDataset::new((0..9).map(|x| x as f64).collect(), 3, 3);
        let rng = StdRng::seed_from_u64(42);
        let mut init = KGeometricPP::new(rng);
        assert!(init.uses_distances());
        let mut cent = make_centroids::<f64>(2, 3);
        init.init(&ds, &mut cent, 2);
        for i in 0..2 {
            for &v in cent.center(i) {
                assert!((0.0..9.0).contains(&v));
            }
        }
    }
}
