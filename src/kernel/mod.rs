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
#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
#[cfg(feature = "parallel")]
pub fn compute_kernel_matrix<D, F, K>(points: &[D], kernel: K) -> Vec<Vec<F>>
where
    D: Send + Sync,
    F: Float + Send + Sync + Copy + Default,
    K: Fn(&D, &D) -> F + Sync,
{
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }

    let mut matrix: Vec<Vec<F>> = vec![vec![F::default(); n]; n];
    matrix.par_iter_mut().enumerate().for_each(|(i, row)| {
        for j in i..n {
            row[j] = kernel(&points[i], &points[j]);
        }
    });

    // Mirror upper triangle into lower triangle in parallel by writing rows and reading columns.
    let matrix_ptr_addr = matrix.as_mut_ptr() as usize;
    rayon::scope(|s| {
        for i in 0..n {
            s.spawn(move |_| {
                let matrix_ptr = matrix_ptr_addr as *mut Vec<F>;
                let row_i = unsafe { &mut *matrix_ptr.add(i) };
                #[allow(clippy::needless_range_loop)]
                for j in 0..i {
                    let row_j = unsafe { &*matrix_ptr.add(j) };
                    row_i[j] = row_j[i];
                }
            });
        }
    });

    matrix
}

#[cfg(not(feature = "parallel"))]
pub fn compute_kernel_matrix<D, F, K>(points: &[D], kernel: K) -> Vec<Vec<F>>
where
    D: Send + Sync,
    F: Float + Send + Sync + Copy + Default,
    K: Fn(&D, &D) -> F + Sync,
{
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }

    let mut matrix: Vec<Vec<F>> = vec![vec![F::default(); n]; n];
    for i in 0..n {
        for j in i..n {
            let val = kernel(&points[i], &points[j]);
            matrix[i][j] = val;
            matrix[j][i] = val;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let km = compute_kernel_matrix(&points, kernel);

        assert_eq!(km.len(), 3);
        assert_eq!(km[0].len(), 3);
        for (i, row) in km.iter().enumerate().take(3) {
            for (j, value) in row.iter().enumerate().take(3) {
                assert!(*value == km[j][i]);
            }
        }
        assert_eq!(km[0][0], 5.0);
        assert_eq!(km[1][1], 25.0);
    }

    #[test]
    fn compute_kernel_matrix_polynomial_degree2() {
        let points = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let poly = PolynomialKernel::new(2, 1.0, 0.0);
        let km = compute_kernel_matrix(&points, |x: &Vec<f64>, y: &Vec<f64>| {
            poly.similarity(x.as_slice(), y.as_slice())
        });

        assert_eq!(km[0][0], 1.0);
        assert_eq!(km[0][1], 0.0);
        assert_eq!(km[1][0], 0.0);
        assert_eq!(km[1][1], 1.0);
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
