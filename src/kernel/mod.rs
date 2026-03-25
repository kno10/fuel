pub mod laplace;
pub mod linear;
pub mod polynomial;
pub mod rational_quadratic;
pub mod rbf;
pub mod sigmoid;

use rayon::prelude::*;

/// Compute a full symmetric kernel matrix for a point set using the given similarity function.
/// The kernel function is expected to be symmetric.
pub fn compute_kernel_matrix<F, K>(points: &[Vec<F>], kernel: K) -> Vec<Vec<F>>
where
    F: Send + Sync + Copy + Default,
    K: Fn(&[F], &[F]) -> F + Sync,
{
    let n = points.len();
    let mut matrix: Vec<Vec<F>> = vec![vec![F::default(); n]; n];

    // compute upper triangle in parallel
    let upper: Vec<Vec<F>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut row = vec![F::default(); n];
            for j in i..n {
                let v = kernel(&points[i], &points[j]);
                row[j] = v;
            }
            row
        })
        .collect();

    for i in 0..n {
        for j in i..n {
            let val = upper[i][j];
            matrix[i][j] = val;
            matrix[j][i] = val;
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_kernel_matrix_symmetry() {
        let points = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let kernel = |x: &[f64], y: &[f64]| x.iter().zip(y.iter()).map(|(a, b)| a * b).sum::<f64>();

        let km = compute_kernel_matrix(&points, kernel);

        assert_eq!(km.len(), 3);
        assert_eq!(km[0].len(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!(km[i][j] == km[j][i]);
            }
        }
        assert_eq!(km[0][0], 5.0);
        assert_eq!(km[1][1], 25.0);
    }

    #[test]
    fn compute_kernel_matrix_polynomial_degree2() {
        let points = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let poly = crate::kernel::polynomial::PolynomialKernel::new(2, 1.0, 0.0);
        let km = compute_kernel_matrix(&points, |x, y| poly.similarity(x, y));

        assert_eq!(km[0][0], 1.0);
        assert_eq!(km[0][1], 0.0);
        assert_eq!(km[1][0], 0.0);
        assert_eq!(km[1][1], 1.0);
    }

    #[test]
    fn kernel_type_smoke_tests() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];

        let lin = crate::kernel::linear::LinearKernel::new(0.0);
        assert_eq!(lin.similarity(&x, &y), 20.0);

        let sig = crate::kernel::sigmoid::SigmoidKernel::new(0.5, 0.0);
        assert!(sig.similarity(&x, &y).abs() <= 1.0);

        let rbf = crate::kernel::rbf::RadialBasisFunctionKernel::new(0.5);
        assert!(rbf.similarity(&x, &y) > 0.0);

        let rat = crate::kernel::rational_quadratic::RationalQuadraticKernel::new(1.0);
        assert!(rat.similarity(&x, &y) <= 1.0);

        let lap = crate::kernel::laplace::LaplaceKernel::new(1.0);
        assert!(lap.similarity(&x, &y) > 0.0);
    }
}
