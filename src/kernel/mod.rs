//! Kernel functions for similarity measurement in clustering/outlier algorithms.
//!
//! Included kernels:
//! - `LinearKernel`: $k(x, y) = \langle x, y\rangle + c$
//! - `PolynomialKernel`: $k(x, y) = (\gamma \langle x, y\rangle + c)^d$
//! - `RadialBasisFunctionKernel`: $k(x, y) = \exp(-\gamma \|x - y\|^2)$
//! - `LaplaceKernel`: $k(x, y) = \exp(-0.5/\sigma^2 \|x - y\|)$
//! - `RationalQuadraticKernel`: $k(x, y) = 1 - \frac{\|x - y\|^2}{\|x - y\|^2 + c}$
//! - `SigmoidKernel`: $k(x, y) = \tanh(c \langle x, y\rangle + \theta)$
//!
//! The library exposes:
//! - `compute_kernel_matrix` for symmetric pairwise kernel matrices, but likely to be moved to a more central location soon.

pub mod laplace;
pub mod linear;
pub mod polynomial;
pub mod rational_quadratic;
pub mod rbf;
pub mod sigmoid;

// `rayon` is only required when the `parallel` feature is enabled.  The
// benchmark disables this feature to measure single-threaded performance.

use crate::Float;

/// Trait for kernel similarity functions.
pub trait Kernel<D: ?Sized, F: Float> {
    fn similarity(&self, x: &D, y: &D) -> F;
}

/// Allow using boxed kernels as trait objects.
impl<D: ?Sized, F: Float, K> Kernel<D, F> for Box<K>
where
    K: Kernel<D, F> + ?Sized,
{
    fn similarity(&self, x: &D, y: &D) -> F { (**self).similarity(x, y) }
}

/// Compute a full symmetric kernel matrix for a point set using the given similarity function.
///
/// The kernel function is expected to be symmetric; diagonal values are computed with the
/// same function (i == j). The result is a square matrix of size n x n, where n = points.len().
///
/// This implementation computes the upper triangle in parallel and mirrors it.
#[cfg(test)]
mod tests {
    use crate::compute_pairwise_dense;
    use crate::kernel::laplace::LaplaceKernel;
    use crate::kernel::linear::LinearKernel;
    use crate::kernel::polynomial::PolynomialKernel;
    use crate::kernel::rational_quadratic::RationalQuadraticKernel;
    use crate::kernel::rbf::RadialBasisFunctionKernel;
    use crate::kernel::sigmoid::SigmoidKernel;

    #[test]
    fn compute_kernel_matrix_symmetry() {
        let points = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let kernel =
            |x: &Vec<f64>, y: &Vec<f64>| x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();

        let km = compute_pairwise_dense(&points, &kernel);

        assert_eq!(km.shape(), [3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(km[[i, j]], km[[j, i]]);
            }
        }
        assert_eq!(km[[0, 0]], 5.0);
        assert_eq!(km[[1, 1]], 25.0);
    }

    #[test]
    fn compute_kernel_matrix_polynomial_degree2() {
        let points = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let poly = PolynomialKernel::new(2, 1.0, 0.0);
        let km = compute_pairwise_dense(&points, &|x: &Vec<f64>, y: &Vec<f64>| {
            poly.similarity(x.as_slice(), y.as_slice())
        });

        assert_eq!(km[[0, 0]], 1.0);
        assert_eq!(km[[0, 1]], 0.0);
        assert_eq!(km[[1, 0]], 0.0);
        assert_eq!(km[[1, 1]], 1.0);
    }

    #[test]
    fn kernel_type_smoke_tests() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];

        let lin = LinearKernel::<f64>::new(0.0);
        assert_eq!(lin.similarity(&x, &y), 20.0);

        let sig = SigmoidKernel::<f64>::new(0.5, 0.0);
        assert!(sig.similarity(&x, &y).abs() <= 1.0);

        let rbf = RadialBasisFunctionKernel::<f64>::new_sigma(0.5);
        assert!(rbf.similarity(&x, &y) > 0.0);

        let rat = RationalQuadraticKernel::<f64>::new(1.0);
        assert!(rat.similarity(&x, &y) <= 1.0);

        let lap = LaplaceKernel::new_sigma(1.0);
        assert!(lap.similarity(&x, &y) > 0.0);
    }
}
